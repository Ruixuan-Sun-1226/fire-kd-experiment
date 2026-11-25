"""One-batch validation smoke test for RGB->IR feature distillation.

Loads one paired validation batch, runs the teacher (IR) and student (RGB)
in eval mode under ``torch.no_grad()``, computes the validation detection
loss and P3/P4/P5 distillation loss, and verifies that no gradients are
produced for either model.  No training or backward pass is performed.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.kd_loss import FeatureDistillationLoss, parse_kd_layers  # noqa: E402
from models.kd_features import FeatureExtractor  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.kd_validation import validation_step  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.paired_dataset import PairedRGBIRDataset, collate_paired  # noqa: E402

DEFAULT_CFG = r'D:\fire_kd_experiment\configs\dcf-yolo-flame2-fire.yaml'
DEFAULT_RGB_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\rgb_fire'
DEFAULT_IR_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\ir_fire'
DEFAULT_HYP = str(REPO_ROOT / 'data' / 'hyp.scratch.tiny.yaml')
DEFAULT_TEACHER_WEIGHTS = \
    r'D:\fire_kd_experiment\runs\smoke_test\ir_fire_epoch1_focaler_iou\weights\best.pt'


def parse_args():
    parser = argparse.ArgumentParser(description='One-batch KD validation test')
    parser.add_argument('--cfg', default=DEFAULT_CFG)
    parser.add_argument('--rgb-root', default=DEFAULT_RGB_ROOT)
    parser.add_argument('--ir-root', default=DEFAULT_IR_ROOT)
    parser.add_argument('--hyp', default=DEFAULT_HYP)
    parser.add_argument('--teacher-weights', default=DEFAULT_TEACHER_WEIGHTS)
    parser.add_argument('--student-weights', default=None)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lambda-feat', type=float, default=1.0)
    parser.add_argument('--kd-layers', default='P3,P4,P5',
                        help='comma-separated distillation layers from P3,P4,P5')
    parser.add_argument('--device', default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    try:
        args.kd_layers = parse_kd_layers(args.kd_layers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weights(model, path, label):
    if not path:
        return
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint
    if hasattr(state, 'float'):
        state = state.float().state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'{label} weights: {path} (missing={len(missing)}, unexpected={len(unexpected)})')


def grad_counts(model):
    """Return (params_with_grad, params_with_nonzero_grad)."""
    with_grad = 0
    nonzero = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        with_grad += 1
        if torch.count_nonzero(param.grad).item() > 0:
            nonzero += 1
    return with_grad, nonzero


def format_shapes(obj):
    if torch.is_tensor(obj):
        return str(tuple(obj.shape))
    if isinstance(obj, (list, tuple)):
        return '[' + ', '.join(format_shapes(o) for o in obj) + ']'
    return repr(obj)


def main():
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device) if args.device else \
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if not args.teacher_weights or not Path(args.teacher_weights).is_file():
        raise FileNotFoundError(f'Teacher weights not found: {args.teacher_weights}')

    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)

    dataset = PairedRGBIRDataset(
        rgb_root=args.rgb_root, ir_root=args.ir_root, split='val',
        imgsz=args.imgsz, augment=False, hyp=hyp, stride=32, single_cls=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_paired)

    print(f'device: {device}')
    print(f'val pairs: {len(dataset)}  imgsz: {args.imgsz}  batch: {args.batch_size}')
    print(f'lambda_feat: {args.lambda_feat}  kd_layers: {",".join(args.kd_layers)}')

    teacher = Model(args.cfg).to(device)
    load_weights(teacher, args.teacher_weights, 'teacher')
    teacher.nc = 1
    teacher.names = ['fire']
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = Model(args.cfg).to(device)
    load_weights(student, args.student_weights, 'student')
    student.nc = 1
    student.names = ['fire']
    student.eval()

    # ComputeLoss needs hyp/gr attached, as in train.py.
    loss_hyp = yaml.safe_load(yaml.dump(hyp))
    nl = student.model[-1].nl
    loss_hyp['box'] *= 3. / nl
    loss_hyp['cls'] *= 1 / 80. * 3. / nl
    loss_hyp['obj'] *= (args.imgsz / 640) ** 2 * 3. / nl
    loss_hyp['label_smoothing'] = 0.0
    student.hyp = loss_hyp
    student.gr = 1.0
    compute_loss = ComputeLoss(student)
    kd_criterion = FeatureDistillationLoss(selected_layers=args.kd_layers)

    teacher_features = FeatureExtractor(teacher)
    student_features = FeatureExtractor(student)

    try:
        batch = next(iter(loader))
        print(f'pair: {batch["image_id"]}')
        print(f'RGB target shape: {tuple(batch["rgb_targets"].shape)}')
        print(f'teacher mode: {"train" if teacher.training else "eval"}  '
              f'student mode: {"train" if student.training else "eval"}')

        # Validation forward only: validation_step is decorated with no_grad.
        result = validation_step(teacher_features, student_features,
                                 compute_loss, kd_criterion, batch, device,
                                 args.lambda_feat)

        teacher_grad_count, _ = grad_counts(teacher)
        student_grad_count, student_nonzero_grad_count = grad_counts(student)

        print(f'val_detection_loss: {result["detection_loss"].item():.6f}')
        print(f'val_feature_loss: {result["feature_loss"].item():.6f}')
        print(f'val_total_loss: {result["total_loss"].item():.6f}')
        for level in args.kd_layers:
            print(f'val_{level.lower()}_loss: '
                  f'{result["layer_losses"][level].item():.6f}')
        print(f'student output shape: {format_shapes(result["student_output"])}')
        print(f'student prediction shape: {tuple(result["student_output"][0].shape)}')
        print(f'teacher_grad_count: {teacher_grad_count}')
        print(f'student_grad_count: {student_grad_count}')
        print(f'student_nonzero_grad_count: {student_nonzero_grad_count}')

        assert torch.isfinite(result['total_loss']).item()
        assert set(result['layer_losses']) == set(args.kd_layers), (
            f'layer_losses keys {sorted(result["layer_losses"])} != '
            f'selected layers {sorted(args.kd_layers)}')
        assert not teacher.training and not student.training
        assert teacher_grad_count == 0, 'teacher produced gradients'
        assert student_grad_count == 0, 'student produced gradients during validation'
        assert student_nonzero_grad_count == 0
        print('kd validation test passed: no backward, no teacher/student gradients')
    finally:
        teacher_features.close()
        student_features.close()


if __name__ == '__main__':
    main()
