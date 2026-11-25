"""Validation helpers for RGB->IR cross-modal feature distillation.

Two independent passes are provided:

* :func:`validation_step` / :func:`validate` compute the RGB detection loss
  and the P3/P4/P5 distillation loss on paired RGB-IR batches;
* :func:`evaluate_rgb_map` computes RGB Precision/Recall/mAP50/mAP50:95
  using the same per-image matching logic as ``test.py`` (no files are
  written and the prepared dataset directory is not modified).

All functions run under ``torch.no_grad()``; callers put both models in
``eval()`` mode beforehand.
"""

import numpy as np
import torch

from utils.general import box_iou, non_max_suppression, xywh2xyxy
from utils.metrics import ap_per_class


def _raw_detection_output(model_output):
    """IDetect returns a list in train mode and (pred, raw) in eval mode."""
    return model_output[1] if isinstance(model_output, tuple) else model_output


@torch.no_grad()
def validation_step(teacher_extractor, student_extractor, compute_loss,
                    kd_criterion, batch, device, lambda_feat):
    """Run one paired validation batch; return losses and student output."""
    rgb = batch['rgb_image'].to(device, non_blocking=True).float() / 255.0
    ir = batch['ir_image'].to(device, non_blocking=True).float() / 255.0
    rgb_targets = batch['rgb_targets'].to(device)

    _, teacher_feats = teacher_extractor(ir)
    student_output, student_feats = student_extractor(rgb)

    detection_loss, _ = compute_loss(
        _raw_detection_output(student_output), rgb_targets)
    feature_loss, layer_losses = kd_criterion(teacher_feats, student_feats)
    total_loss = detection_loss + lambda_feat * feature_loss

    return {
        'detection_loss': detection_loss,
        'feature_loss': feature_loss,
        'total_loss': total_loss,
        'layer_losses': layer_losses,
        'student_output': student_output,
    }


@torch.no_grad()
def validate(teacher_extractor, student_extractor, compute_loss,
             kd_criterion, loader, device, lambda_feat):
    """Average validation losses over a paired RGB-IR loader.

    Per-level sums are created only for the layers selected by
    ``kd_criterion`` so an ablation run reports just the distilled layers.
    """
    sums = {'detection_loss': 0.0, 'feature_loss': 0.0, 'total_loss': 0.0}
    for level in kd_criterion.selected_layers:
        sums[f'{level.lower()}_loss'] = 0.0

    for batch in loader:
        result = validation_step(teacher_extractor, student_extractor,
                                 compute_loss, kd_criterion, batch, device,
                                 lambda_feat)
        sums['detection_loss'] += result['detection_loss'].item()
        sums['feature_loss'] += result['feature_loss'].item()
        sums['total_loss'] += result['total_loss'].item()
        for level, level_loss in result['layer_losses'].items():
            sums[f'{level.lower()}_loss'] += level_loss.item()

    n = max(len(loader), 1)
    return {key: value / n for key, value in sums.items()}


@torch.no_grad()
def evaluate_rgb_map(model, loader, device, nc=1, conf_thres=0.001,
                     iou_thres=0.6, multi_label=True):
    """RGB-only P/R/mAP, mirroring test.py's per-image matching.

    Predictions and targets are both kept in the letterboxed model-input
    coordinate system, so the native-space rescaling done by test.py (which
    applies the same transform to predictions and targets) is unnecessary.
    """
    was_training = model.training
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    stats = []
    seen = 0

    for batch in loader:
        imgs = batch['rgb_image'].to(device, non_blocking=True).float() / 255.0
        targets = batch['rgb_targets'].to(device)
        _, _, height, width = imgs.shape

        output = model(imgs)
        predictions = output[0] if isinstance(output, tuple) else output

        targets[:, 2:] *= torch.tensor([width, height, width, height],
                                       device=device, dtype=targets.dtype)
        predictions = non_max_suppression(
            predictions, conf_thres=conf_thres, iou_thres=iou_thres,
            multi_label=multi_label)

        for si, pred in enumerate(predictions):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((
                        torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(), torch.Tensor(), tcls,
                    ))
                continue

            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool,
                                  device=device)
            if nl:
                detected = set()
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    if pi.shape[0]:
                        ious, indices = box_iou(pred[pi, :4], tbox[ti]).max(1)
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            matched = ti[indices[j]]
                            if matched.item() not in detected:
                                detected.add(matched.item())
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(),
                          pred[:, 5].cpu(), tcls))

    if was_training:
        model.train()

    results = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0,
               'mAP50_95': 0.0, 'images': seen}
    if not stats:
        return results

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, _, _ = ap_per_class(*stats, plot=False, save_dir='',
                                      names={})
        ap50 = ap[:, 0]
        results.update(
            precision=float(p.mean()),
            recall=float(r.mean()),
            mAP50=float(ap50.mean()),
            mAP50_95=float(ap.mean()),
        )
    return results
