"""Unit test for ``--kd-layers`` selection in FeatureDistillationLoss.

Uses synthetic feature maps only, so it runs without the prepared dataset or
teacher weights.  Verifies:

* P3-only / P4-only / P5-only return exactly that level's loss;
* P3,P4,P5 returns the full loss (sum of all three, weights 1.0);
* unselected levels are never read, so they may be absent from the dicts;
* gradients flow only into the selected level's student feature.

Run from anywhere::

    python scripts/test_kd_layers.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.kd_loss import (  # noqa: E402
    FeatureDistillationLoss,
    parse_kd_layers,
)

LEVEL_SHAPES = {'P3': (2, 16, 8, 8), 'P4': (2, 32, 4, 4), 'P5': (2, 64, 2, 2)}


def make_features(levels, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    feats = {}
    for level in levels:
        feat = torch.randn(*LEVEL_SHAPES[level])
        if requires_grad:
            feat = feat.requires_grad_(True)
        feats[level] = feat
    return feats


def test_parse_layers():
    assert parse_kd_layers(None) == ['P3', 'P4', 'P5']
    assert parse_kd_layers('P3') == ['P3']
    assert parse_kd_layers('P5,P3') == ['P3', 'P5']  # canonical order
    assert parse_kd_layers('p3,p3') == ['P3']  # case-insensitive, deduped
    for invalid in ('P2', 'P3,P6', '', ' , '):
        try:
            parse_kd_layers(invalid)
        except ValueError:
            continue
        raise AssertionError(f'parse_kd_layers accepted invalid value {invalid!r}')
    print('parse_kd_layers: ok')


def test_single_layer_selection():
    for level in ('P3', 'P4', 'P5'):
        teacher = make_features([level], seed=1)
        student = make_features([level], seed=2)
        criterion = FeatureDistillationLoss(selected_layers=level)
        total, layer_losses = criterion(teacher, student)

        assert criterion.selected_layers == [level]
        assert set(layer_losses) == {level}, (
            f'{level}-only returned {sorted(layer_losses)}')
        assert torch.allclose(total, layer_losses[level]), \
            f'{level}-only total is not the {level} loss'
        assert torch.isfinite(total).item()
        print(f'{level}-only: keys={sorted(layer_losses)} '
              f'loss={layer_losses[level].item():.6f} total={total.item():.6f}')


def test_all_layers_selection():
    levels = ['P3', 'P4', 'P5']
    teacher = make_features(levels, seed=3)
    student = make_features(levels, seed=4)
    criterion = FeatureDistillationLoss()  # default: all three, weight 1.0
    total, layer_losses = criterion(teacher, student)

    assert criterion.selected_layers == levels
    assert set(layer_losses) == set(levels)
    expected = sum(layer_losses[level] for level in levels)
    assert torch.allclose(total, expected, atol=1e-6), \
        'P3,P4,P5 total != sum of the three unit-weight losses'
    print('P3,P4,P5: ' + ' '.join(
        f'{level}={layer_losses[level].item():.6f}' for level in levels)
        + f' total={total.item():.6f}')


def test_unselected_levels_are_not_read():
    # Only P4 is present in the dictionaries, but the selection is P4 too.
    teacher = make_features(['P4'], seed=5)
    student = make_features(['P4'], seed=6)
    criterion = FeatureDistillationLoss(selected_layers='P4')
    total, layer_losses = criterion(teacher, student)
    assert set(layer_losses) == {'P4'}
    assert torch.isfinite(total).item()
    print('unselected levels not read: ok')


def test_gradients_only_on_selected_level():
    teacher = make_features(['P3', 'P4', 'P5'], seed=7)
    student = make_features(['P3', 'P4', 'P5'], requires_grad=True, seed=8)
    criterion = FeatureDistillationLoss(selected_layers='P5')
    total, _ = criterion(teacher, student)
    total.backward()

    grads = {level: student[level].grad is not None for level in student}
    assert grads == {'P3': False, 'P4': False, 'P5': True}, grads
    print('gradients only on P5: ok')


def main():
    test_parse_layers()
    test_single_layer_selection()
    test_all_layers_selection()
    test_unselected_levels_are_not_read()
    test_gradients_only_on_selected_level()
    print('all kd-layer selection tests passed')


if __name__ == '__main__':
    main()
