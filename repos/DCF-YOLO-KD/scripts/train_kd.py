"""Minimal cross-modal feature-distillation training for RGB student / IR teacher.

Pipeline:
* paired RGB-IR training and validation datasets (paired by file name);
* RGB labels drive the original detection loss (``ComputeLoss``);
* IR teacher is eval() and frozen;
* P3/P4/P5 feature distillation with fixed layer weights (1, 1, 1);
* total_loss = detection_loss + lambda_feat * feature_loss;
* each epoch: train, validate losses, RGB P/R/mAP50/mAP50:95, save checkpoints;
* best.pt is selected by val mAP50:95 only (never by val_total_loss);
  last.pt always holds the final epoch;
* no scale adaptation, box matching, uncertainty weighting, or hard negatives.

Paths are command-line arguments; the defaults only reflect the current
local experiment layout.  The original DCF-YOLO inference interface and the
prepared dataset are not modified.
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.kd_loss import FeatureDistillationLoss, parse_kd_layers  # noqa: E402
from models.kd_features import FeatureExtractor  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.kd_validation import evaluate_rgb_map, validate  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.paired_dataset import PairedRGBIRDataset, collate_paired  # noqa: E402

DEFAULT_CFG = r'D:\fire_kd_experiment\configs\dcf-yolo-flame2-fire.yaml'
DEFAULT_RGB_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\rgb_fire'
DEFAULT_IR_ROOT = r'D:\fire_kd_experiment\datasets\FLAME2-DT\prepared\ir_fire'
DEFAULT_HYP = str(REPO_ROOT / 'data' / 'hyp.scratch.tiny.yaml')
DEFAULT_TEACHER_WEIGHTS = \
    r'D:\fire_kd_experiment\runs\smoke_test\ir_fire_epoch1_focaler_iou\weights\best.pt'
DEFAULT_SAVE_DIR = \
    r'D:\fire_kd_experiment\runs\kd_smoke\rgb_student_ir_teacher'

# best.pt is selected on validation mAP50:95 only; val_total_loss is never
# used as the checkpoint criterion.
SELECTION_METRIC = 'val_mAP50_95'


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal RGB->IR feature KD training')
    parser.add_argument('--cfg', default=DEFAULT_CFG, help='model YAML')
    parser.add_argument('--rgb-root', default=DEFAULT_RGB_ROOT, help='RGB dataset root')
    parser.add_argument('--ir-root', default=DEFAULT_IR_ROOT, help='IR dataset root')
    parser.add_argument('--hyp', default=DEFAULT_HYP, help='augmentation/loss hyp YAML')
    parser.add_argument('--teacher-weights', default=DEFAULT_TEACHER_WEIGHTS,
                        help='IR teacher checkpoint')
    parser.add_argument('--student-weights', default=None,
                        help='optional RGB student checkpoint (random init if omitted)')
    parser.add_argument('--save-dir', default=DEFAULT_SAVE_DIR,
                        help='directory for last.pt/best.pt and run metadata')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='', help='e.g. cuda:0; default cuda:0 if available')
    parser.add_argument('--lambda-feat', type=float, default=1.0)
    parser.add_argument('--kd-layers', default='P3,P4,P5',
                        help='comma-separated distillation layers from P3,P4,P5 '
                             '(e.g. "P3", "P3,P4", "P4,P5"); default all three')
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


def scaled_hyp(hyp, model, imgsz, nc):
    """Apply the same loss-gain scaling as train.py."""
    hyp = copy.deepcopy(hyp)
    nl = model.model[-1].nl
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = 0.0
    return hyp


def build_optimizer(model, hyp):
    pg0, pg1, pg2 = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.bias' in name:
            pg2.append(param)
        elif '.weight' in name and param.ndim >= 2:
            pg1.append(param)
        else:
            pg0.append(param)
    return optim.SGD(
        [
            {'params': pg0},
            {'params': pg1, 'weight_decay': hyp['weight_decay']},
            {'params': pg2},
        ],
        lr=hyp['lr0'],
        momentum=hyp['momentum'],
        nesterov=True,
    )


def select_best(val_map, best_selection_value):
    """Return ``(selection_value, selected)`` for best.pt selection.

    Selection is driven exclusively by validation mAP50:95 and keeps the
    first epoch (``best_selection_value`` starts at ``-inf``).
    """
    selection_value = float(val_map['mAP50_95'])
    return selection_value, selection_value > best_selection_value


def save_checkpoint(path, model, optimizer, args, epoch, best_epoch,
                    selection_metric, selection_value,
                    train_losses, val_losses, val_map,
                    best_map50, best_map50_95):
    """Save an attempt_load-compatible checkpoint with KD metadata.

    ``epoch`` and ``best_epoch`` are 0-based indices, matching the ``epoch``
    field used by ``train.py`` checkpoints.  ``selection_value`` is the
    mAP50:95 of ``epoch``; for best.pt it therefore equals the value that
    triggered the save, and ``epoch == best_epoch``.
    """
    checkpoint = {
        'model': copy.deepcopy(model).half(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_epoch': best_epoch,
        'selection_metric': selection_metric,
        'selection_value': float(selection_value),
        'nc': int(getattr(model, 'nc', 1)),
        'names': list(getattr(model, 'names', ['fire'])),
        'train_losses': dict(train_losses),
        'val_losses': dict(val_losses),
        'val_map': dict(val_map),
        'best_mAP50': float(best_map50),
        'best_mAP50_95': float(best_map50_95),
        'lambda_feat': args.lambda_feat,
        'kd_layers': list(getattr(args, 'kd_layers', ('P3', 'P4', 'P5'))),
        'teacher_weights': args.teacher_weights,
        'student_weights': args.student_weights,
        'training_results': None,
        'ema': None,
        'updates': None,
        'wandb_id': None,
        'kd_args': vars(args),
    }
    torch.save(checkpoint, path)


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

    train_dataset = PairedRGBIRDataset(
        rgb_root=args.rgb_root, ir_root=args.ir_root, split='train',
        imgsz=args.imgsz, augment=True, hyp=hyp, stride=32, single_cls=True)
    val_dataset = PairedRGBIRDataset(
        rgb_root=args.rgb_root, ir_root=args.ir_root, split='val',
        imgsz=args.imgsz, augment=False, hyp=hyp, stride=32, single_cls=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        collate_fn=collate_paired)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        collate_fn=collate_paired)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / 'last.pt'
    best_path = save_dir / 'best.pt'
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    print(f'device: {device}')
    print(f'train pairs: {len(train_dataset)}  batches/epoch: {len(train_loader)}')
    print(f'val pairs:   {len(val_dataset)}  batches/epoch: {len(val_loader)}')
    print(f'epochs: {args.epochs}  batch: {args.batch_size}  imgsz: {args.imgsz}  '
          f'lambda_feat: {args.lambda_feat}  '
          f'kd_layers: {",".join(args.kd_layers)}')
    print(f'save dir: {save_dir}')

    # Teacher: IR, pretrained, eval + frozen for the whole run.
    teacher = Model(args.cfg).to(device)
    load_weights(teacher, args.teacher_weights, 'teacher')
    teacher.nc = 1
    teacher.names = ['fire']
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # Student: RGB, trainable.
    student = Model(args.cfg).to(device)
    load_weights(student, args.student_weights, 'student')
    student.nc = 1
    student.names = ['fire']
    student.train()

    loss_hyp = scaled_hyp(hyp, student, args.imgsz, nc=1)
    student.hyp = loss_hyp
    student.gr = 1.0
    compute_loss = ComputeLoss(student)
    kd_criterion = FeatureDistillationLoss(selected_layers=args.kd_layers)
    optimizer = build_optimizer(student, hyp)

    teacher_features = FeatureExtractor(teacher)
    student_features = FeatureExtractor(student)

    best_selection_value = -float('inf')
    best_epoch = -1
    best_map50 = 0.0
    best_map50_95 = 0.0
    history = []

    try:
        for epoch in range(args.epochs):
            # ---------------------------- train ----------------------------
            student.train()
            train_sums = {'detection_loss': 0.0, 'feature_loss': 0.0,
                          'total_loss': 0.0}
            for level in kd_criterion.selected_layers:
                train_sums[f'{level.lower()}_loss'] = 0.0

            for i, batch in enumerate(train_loader):
                rgb = batch['rgb_image'].to(device, non_blocking=True).float() / 255.0
                ir = batch['ir_image'].to(device, non_blocking=True).float() / 255.0
                rgb_targets = batch['rgb_targets'].to(device)

                with torch.no_grad():
                    _, teacher_feats = teacher_features(ir)

                student_output, student_feats = student_features(rgb)
                detection_loss, _ = compute_loss(student_output, rgb_targets)
                feature_loss, layer_losses = kd_criterion(teacher_feats, student_feats)
                total_loss = detection_loss + args.lambda_feat * feature_loss

                if not torch.isfinite(total_loss).item():
                    raise FloatingPointError(
                        f'Non-finite train total_loss at epoch {epoch} batch {i}: '
                        f'{total_loss.item()}')

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                train_sums['detection_loss'] += detection_loss.item()
                train_sums['feature_loss'] += feature_loss.item()
                train_sums['total_loss'] += total_loss.item()
                for level, level_loss in layer_losses.items():
                    train_sums[f'{level.lower()}_loss'] += level_loss.item()

                if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                    print(f'  epoch {epoch + 1}/{args.epochs} '
                          f'train batch {i + 1}/{len(train_loader)} '
                          f'total_loss={total_loss.item():.6f}')

            n_train = max(len(train_loader), 1)
            train_losses = {k: v / n_train for k, v in train_sums.items()}

            # --------------------------- validate --------------------------
            student.eval()
            val_losses = validate(teacher_features, student_features,
                                  compute_loss, kd_criterion, val_loader,
                                  device, args.lambda_feat)
            val_map = evaluate_rgb_map(student, val_loader, device, nc=1)

            student.train()

            selection_value, selected_best = select_best(
                val_map, best_selection_value)
            if selected_best:
                best_selection_value = selection_value
                best_epoch = epoch

            best_map50 = max(best_map50, val_map['mAP50'])
            best_map50_95 = max(best_map50_95, val_map['mAP50_95'])

            record = {
                'epoch': epoch + 1,
                'train_detection_loss': train_losses['detection_loss'],
                'train_feature_loss': train_losses['feature_loss'],
                'train_total_loss': train_losses['total_loss'],
                'val_detection_loss': val_losses['detection_loss'],
                'val_feature_loss': val_losses['feature_loss'],
                'val_total_loss': val_losses['total_loss'],
                'val_precision': val_map['precision'],
                'val_recall': val_map['recall'],
                'val_mAP50': val_map['mAP50'],
                'val_mAP50_95': val_map['mAP50_95'],
                'selection_value': selection_value,
            }
            history.append(record)

            layer_msg = ' '.join(
                f'train_{level.lower()}_loss={train_losses[level.lower() + "_loss"]:.6f}'
                for level in kd_criterion.selected_layers)
            print(
                f'epoch {epoch + 1}/{args.epochs} '
                f'kd_layers={",".join(kd_criterion.selected_layers)} '
                f'train_detection_loss={record["train_detection_loss"]:.6f} '
                f'train_feature_loss={record["train_feature_loss"]:.6f} '
                f'{layer_msg} '
                f'train_total_loss={record["train_total_loss"]:.6f} '
                f'val_detection_loss={record["val_detection_loss"]:.6f} '
                f'val_feature_loss={record["val_feature_loss"]:.6f} '
                f'val_total_loss={record["val_total_loss"]:.6f} '
                f'val_precision={record["val_precision"]:.6f} '
                f'val_recall={record["val_recall"]:.6f} '
                f'val_mAP50={record["val_mAP50"]:.6f} '
                f'val_mAP50_95={record["val_mAP50_95"]:.6f} '
                f'selection_metric={SELECTION_METRIC} '
                f'selection_value={selection_value:.6f}'
            )

            # Hooks hold unpicklable closures; remove them before saving.
            student_features.close()
            try:
                save_checkpoint(last_path, student, optimizer, args, epoch,
                                best_epoch, SELECTION_METRIC, selection_value,
                                train_losses, val_losses, val_map,
                                best_map50, best_map50_95)
                if selected_best:
                    save_checkpoint(best_path, student, optimizer, args, epoch,
                                    best_epoch, SELECTION_METRIC,
                                    selection_value,
                                    train_losses, val_losses, val_map,
                                    best_map50, best_map50_95)
                    print(f'  saved improved best.pt '
                          f'(epoch={best_epoch} '
                          f'{SELECTION_METRIC}={best_selection_value:.6f})')
                print(f'  selected_best_epoch={best_epoch} '
                      f'best_selection_value={best_selection_value:.6f} '
                      f'best_mAP50={best_map50:.6f} '
                      f'best_mAP50_95={best_map50_95:.6f}')
                print(f'  saved last.pt -> {last_path}')
            finally:
                student_features = FeatureExtractor(student)

        print('training finished; per-epoch results:')
        for record in history:
            print('  ' + ' '.join(f'{k}={v:.6f}' for k, v in record.items()
                                   if k != 'epoch') + f' (epoch {record["epoch"]})')
        print(f'selected_best_epoch={best_epoch}  '
              f'best_selection_value={best_selection_value:.6f}  '
              f'best_mAP50={best_map50:.6f}  '
              f'best_mAP50_95={best_map50_95:.6f}  '
              f'selection_metric={SELECTION_METRIC}')
    finally:
        teacher_features.close()
        student_features.close()


if __name__ == '__main__':
    main()
