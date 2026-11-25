"""Feature-map knowledge distillation loss for cross-modal detection.

The loss follows a plain, non-adaptive formulation:

1. for each level (P3/P4/P5), L2-normalize every spatial feature vector
   across its channel dimension;
2. compute MSE between the normalized student and teacher maps;
3. return the weighted sum plus the individual (unweighted) level losses.

Layer weights default to P3=1.0, P4=1.0, P5=1.0.  No scale adaptation,
boundary matching, uncertainty weighting, or hard-negative sampling is
included.

``selected_layers`` restricts distillation to a subset of P3/P4/P5 (used by
the single-layer ablation); layers outside the selection are not read from
the feature dictionaries and contribute nothing to the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_LAYERS = ('P3', 'P4', 'P5')
DEFAULT_SELECTED_LAYERS = ('P3', 'P4', 'P5')
DEFAULT_LAYER_WEIGHTS = {'P3': 1.0, 'P4': 1.0, 'P5': 1.0}


def parse_kd_layers(value):
    """Normalize a layer selection into an ordered ``list`` of P3/P4/P5.

    Args:
        value: ``None`` (all layers), a comma-separated string such as
            ``'P3,P4'``, or any iterable of layer names.  Case-insensitive.

    Returns:
        Layers in canonical P3, P4, P5 order, duplicates removed.
    """
    if value is None:
        return list(DEFAULT_SELECTED_LAYERS)
    if isinstance(value, str):
        raw = value.split(',')
    else:
        raw = list(value)

    selected = set()
    for item in raw:
        name = str(item).strip().upper()
        if not name:
            continue
        if name not in VALID_LAYERS:
            raise ValueError(
                f'Unknown distillation layer {item!r}; '
                f'expected one of {", ".join(VALID_LAYERS)}')
        selected.add(name)
    if not selected:
        raise ValueError(
            'kd_layers must select at least one of '
            f'{", ".join(VALID_LAYERS)}')
    return [level for level in VALID_LAYERS if level in selected]


class FeatureDistillationLoss(nn.Module):
    """MSE between channel-normalized student and teacher feature maps."""

    def __init__(self, layer_weights=None, detach_teacher=True, eps=1e-12,
                 selected_layers=None):
        super().__init__()
        weights = dict(DEFAULT_LAYER_WEIGHTS)
        if layer_weights is not None:
            unknown = set(layer_weights) - set(weights)
            if unknown:
                raise KeyError(f'Unknown feature levels: {sorted(unknown)}')
            weights.update(layer_weights)
        self.layer_weights = weights
        self.selected_layers = parse_kd_layers(selected_layers)
        self.detach_teacher = detach_teacher
        self.eps = eps
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, teacher_features, student_features):
        """Return (weighted_total, {level: unweighted_mse}).

        Args:
            teacher_features: dict with at least the selected keys
                (B, C, H, W).
            student_features: dict with the same keys and shapes.
        """
        layer_losses = {}
        total = None

        for level in self.selected_layers:
            weight = self.layer_weights[level]
            if level not in teacher_features or level not in student_features:
                raise KeyError(f'Missing "{level}" in teacher/student features')
            teacher_feat = teacher_features[level]
            student_feat = student_features[level]

            if teacher_feat.shape != student_feat.shape:
                raise ValueError(
                    f'{level} shape mismatch: teacher {tuple(teacher_feat.shape)} '
                    f'vs student {tuple(student_feat.shape)}'
                )
            if self.detach_teacher:
                teacher_feat = teacher_feat.detach()

            # L2-normalize each spatial location across channels.
            teacher_norm = F.normalize(teacher_feat.float(), p=2, dim=1, eps=self.eps)
            student_norm = F.normalize(student_feat.float(), p=2, dim=1, eps=self.eps)

            level_loss = self.mse(student_norm, teacher_norm)
            layer_losses[level] = level_loss

            weighted = weight * level_loss
            total = weighted if total is None else total + weighted

        return total, layer_losses
