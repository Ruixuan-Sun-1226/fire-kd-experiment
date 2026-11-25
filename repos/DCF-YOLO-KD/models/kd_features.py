"""Intermediate-feature access for cross-modal knowledge distillation.

``Model.forward`` (``models/yolo.py``) returns only the detector output; it
does not expose backbone feature maps.  Rather than changing the inference
signature of the detector (which ``detect.py``/``test.py``/ONNX export all
rely on), this module wraps a model with forward hooks that capture the
backbone P3/P4/P5 tensors *without* modifying any model code.

Layer indices refer to this project's ``dcf-yolo-flame2-fire.yaml``:

    layer 14 -> P3/8  (128 channels)
    layer 21 -> P4/16 (256 channels)
    layer 28 -> P5/32 (512 channels)

Pass a custom mapping if a different YAML is used.

Usage::

    extractor = FeatureExtractor(model)
    output, features = extractor(x)
    p3, p4, p5 = features['P3'], features['P4'], features['P5']
    ...
    extractor.close()

The hooks are registered only for the lifetime of the extractor.  Features
are not detached by default (so a student can backprop through them); pass
``detach=True`` for a frozen teacher if desired (though ``torch.no_grad()``
already detaches).
"""

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

# Backbone outputs of configs/dcf-yolo-flame2-fire.yaml.
DEFAULT_FEATURE_LAYERS = {'P3': 14, 'P4': 21, 'P5': 28}


class FeatureExtractor(nn.Module):
    """Run a YOLO ``Model`` and return its detector output plus P3/P4/P5."""

    def __init__(self, model: nn.Module, feature_layers: Optional[Mapping[str, int]] = None,
                 detach: bool = False):
        super().__init__()
        self.model = model
        self.detach = detach
        self.feature_layers = dict(DEFAULT_FEATURE_LAYERS if feature_layers is None else feature_layers)
        self._features: Dict[str, torch.Tensor] = {}
        self._handles = []

        if not hasattr(model, 'model'):
            raise TypeError('FeatureExtractor expects a models.yolo.Model instance')
        n_layers = len(model.model)
        for name, layer_index in self.feature_layers.items():
            if not 0 <= layer_index < n_layers:
                raise IndexError(
                    f'Feature layer {name}={layer_index} is out of range for a '
                    f'{n_layers}-layer model; check feature_layers.'
                )
            handle = model.model[layer_index].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._features[name] = output.detach() if self.detach else output
        return hook

    def forward(self, x):
        self._features = {}
        output = self.model(x)
        missing = [name for name in self.feature_layers if name not in self._features]
        if missing:
            raise RuntimeError(f'Forward hooks did not fire for features: {missing}')
        features = {name: self._features[name] for name in self.feature_layers}
        return output, features

    def close(self):
        """Remove hooks. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._features = {}

    def __del__(self):
        self.close()
