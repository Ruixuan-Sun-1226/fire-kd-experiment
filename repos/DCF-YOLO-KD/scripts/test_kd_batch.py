"""Single-batch forward smoke test for paired RGB/IR distillation.

This script only validates the data path and the P3/P4/P5 feature hooks:

* teacher (IR input) is ``eval()`` and every parameter is frozen;
* student (RGB input) stays in ``train()``;
* exactly one batch is loaded and passed forward;
* no loss, no ``backward()``, no optimizer step.

Run from anywhere:

    python scripts/test_kd_batch.py
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

from models.kd_features import FeatureExtractor  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.paired_dataset import PairedRGBIRDataset, collate_paired  # noqa: E402

DEFAULT_CFG = r'D:\fire_kd_experiment\configs\dcf-yolo-flame2-fire.yaml'
DEFAULT_RGB_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\rgb_fire'
DEFAULT_IR_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\ir_fire'
DEFAULT_HYP = str(REPO_ROOT / 'data' / 'hyp.scratch.tiny.yaml')


def parse_args():
    parser = argparse.ArgumentParser(description='One-batch RGB/IR KD forward test')
    parser.add_argument('--cfg', default=DEFAULT_CFG, help='model YAML')
    parser.add_argument('--rgb-root', default=DEFAULT_RGB_ROOT, help='RGB dataset root')
    parser.add_argument('--ir-root', default=DEFAULT_IR_ROOT, help='IR dataset root')
    parser.add_argument('--hyp', default=DEFAULT_HYP, help='augmentation hyperparameters')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='', help="e.g. cuda:0; default cuda:0 if available")
    parser.add_argument('--no-augment', action='store_true', help='disable augmentation')
    parser.add_argument('--teacher-weights', default=None)
    parser.add_argument('--student-weights', default=None)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weights(model, path):
    if not path:
        return
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint
    if hasattr(state, 'state_dict'):
        state = state.float().state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'loaded weights {path} (missing={len(missing)}, unexpected={len(unexpected)})')


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_params(model):
    return sum(p.numel() for p in model.parameters())


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

    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)

    augment = not args.no_augment
    dataset = PairedRGBIRDataset(
        rgb_root=args.rgb_root,
        ir_root=args.ir_root,
        split=args.split,
        imgsz=args.imgsz,
        augment=augment,
        hyp=hyp,
        stride=32,
        single_cls=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paired,
    )

    print(f'device: {device}')
    print(f'config: {args.cfg}')
    print(f'RGB root: {args.rgb_root}')
    print(f'IR root:  {args.ir_root}')
    print(f'split: {args.split}  augment: {augment}  imgsz: {args.imgsz}')
    print(f'paired samples: {len(dataset)}')

    # Teacher: IR input, eval mode, all parameters frozen.
    teacher = Model(args.cfg).to(device)
    load_weights(teacher, args.teacher_weights)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Student: RGB input, kept in training mode.
    student = Model(args.cfg).to(device)
    load_weights(student, args.student_weights)
    student.train()

    teacher_features = FeatureExtractor(teacher)
    student_features = FeatureExtractor(student)

    try:
        batch = next(iter(loader))
        rgb = batch['rgb_image'].to(device).float() / 255.0
        ir = batch['ir_image'].to(device).float() / 255.0
        rgb_targets = batch['rgb_targets'].to(device)
        ir_targets = batch['ir_targets'].to(device)

        print(f'paired image_ids: {batch["image_id"]}')
        print(f'teacher mode: {"train" if teacher.training else "eval"}  '
              f'student mode: {"train" if student.training else "eval"}')

        # Forward only: no loss, no backward, no optimizer.
        with torch.no_grad():
            teacher_output, teacher_feats = teacher_features(ir)
        with torch.no_grad():
            student_output, student_feats = student_features(rgb)

        print(f'RGB batch shape: {tuple(rgb.shape)}')
        print(f'IR batch shape: {tuple(ir.shape)}')
        print(f'RGB target shape: {tuple(rgb_targets.shape)}')
        print(f'IR target shape: {tuple(ir_targets.shape)}')
        print(f'teacher output shape: {format_shapes(teacher_output)}')
        print(f'student output shape: {format_shapes(student_output)}')
        for level in ('P3', 'P4', 'P5'):
            print(f'teacher {level} feature shape: {tuple(teacher_feats[level].shape)}')
        for level in ('P3', 'P4', 'P5'):
            print(f'student {level} feature shape: {tuple(student_feats[level].shape)}')
        print(f'trainable teacher parameter count: {trainable_params(teacher)}')
        print(f'trainable student parameter count: {trainable_params(student)}')
        print(f'total teacher parameters: {total_params(teacher)}')
        print(f'total student parameters: {total_params(student)}')

        # Invariants that make the smoke test meaningful.
        assert not teacher.training, 'teacher must be in eval() mode'
        assert student.training, 'student must remain in train() mode'
        assert trainable_params(teacher) == 0, 'teacher parameters must be frozen'
        assert trainable_params(student) > 0, 'student must have trainable parameters'
        for level, stride in (('P3', 8), ('P4', 16), ('P5', 32)):
            t_feat, s_feat = teacher_feats[level], student_feats[level]
            expected = (args.batch_size, t_feat.shape[1],
                        args.imgsz // stride, args.imgsz // stride)
            assert tuple(t_feat.shape) == expected, \
                f'teacher {level} shape {tuple(t_feat.shape)} != {expected}'
            assert tuple(t_feat.shape) == tuple(s_feat.shape), \
                f'{level} teacher/student feature shapes differ'
        print('smoke test passed: paired batch + teacher/student P3/P4/P5 features')
    finally:
        teacher_features.close()
        student_features.close()


if __name__ == '__main__':
    main()
