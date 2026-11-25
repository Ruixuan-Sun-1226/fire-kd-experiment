"""Checkpoint audit test for best.pt selection.

Replays the epoch loop of ``scripts/train_kd.py`` with synthetic validation
metrics and saves real ``last.pt``/``best.pt`` files through
``train_kd.save_checkpoint``.  It then reloads ``best.pt`` and verifies that
its ``val_map`` is the one produced by the mAP50:95-selected epoch -- never
the epoch with the lowest ``val_total_loss``.

No dataset, pretrained weights, GPU, or forward pass is used.

Run from anywhere:

    python scripts/test_kd_checkpoint_audit.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_kd  # noqa: E402

REQUIRED_BEST_KEYS = {
    'epoch',
    'best_epoch',
    'selection_metric',
    'selection_value',
    'val_map',
    'val_losses',
    'best_mAP50',
    'best_mAP50_95',
    'kd_layers',
    'lambda_feat',
    'teacher_weights',
    'student_weights',
}

# Three synthetic epochs:
# * epoch 1 has the highest mAP50:95 but not the lowest val_total_loss;
# * epoch 2 has the lowest val_total_loss (and the highest mAP50) but a
#   lower mAP50:95 than epoch 1, so it must NOT win best.pt.
EPOCHS = [
    {
        'val_map': {'precision': 0.10, 'recall': 0.20,
                    'mAP50': 0.10, 'mAP50_95': 0.10, 'images': 4},
        'val_losses': {'detection_loss': 0.80, 'feature_loss': 0.20,
                       'total_loss': 1.00},
    },
    {
        'val_map': {'precision': 0.40, 'recall': 0.50,
                    'mAP50': 0.50, 'mAP50_95': 0.30, 'images': 4},
        'val_losses': {'detection_loss': 0.90, 'feature_loss': 0.30,
                       'total_loss': 1.20},
    },
    {
        'val_map': {'precision': 0.55, 'recall': 0.60,
                    'mAP50': 0.70, 'mAP50_95': 0.20, 'images': 4},
        'val_losses': {'detection_loss': 0.40, 'feature_loss': 0.10,
                       'total_loss': 0.50},
    },
]
EXPECTED_BEST_EPOCH = 1
EXPECTED_LOSS_MIN_EPOCH = 2
KD_LAYERS = ('P3', 'P4', 'P5')
LAMBDA_FEAT = 0.5
TRAIN_LOSSES = {'detection_loss': 0.7, 'feature_loss': 0.2, 'total_loss': 0.9}


def build_model():
    """Small picklable/half-able stand-in for the DCF-YOLO student."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )
    model.nc = 1
    model.names = ['fire']
    return model


def replay_epochs(save_dir):
    """Mirror train_kd.main()'s selection/save block for the fake epochs."""
    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = SimpleNamespace(
        lambda_feat=LAMBDA_FEAT,
        kd_layers=KD_LAYERS,
        teacher_weights='teacher.pt',
        student_weights=None,
    )
    last_path = save_dir / 'last.pt'
    best_path = save_dir / 'best.pt'

    best_selection_value = -float('inf')
    best_epoch = -1
    best_map50 = 0.0
    best_map50_95 = 0.0
    selected_epochs = []

    for epoch, sample in enumerate(EPOCHS):
        val_map = sample['val_map']
        val_losses = sample['val_losses']

        selection_value, selected_best = train_kd.select_best(
            val_map, best_selection_value)
        if selected_best:
            best_selection_value = selection_value
            best_epoch = epoch
            selected_epochs.append(epoch)

        best_map50 = max(best_map50, val_map['mAP50'])
        best_map50_95 = max(best_map50_95, val_map['mAP50_95'])

        train_kd.save_checkpoint(
            last_path, model, optimizer, args, epoch, best_epoch,
            train_kd.SELECTION_METRIC, selection_value,
            TRAIN_LOSSES, val_losses, val_map, best_map50, best_map50_95)
        if selected_best:
            train_kd.save_checkpoint(
                best_path, model, optimizer, args, epoch, best_epoch,
                train_kd.SELECTION_METRIC, selection_value,
                TRAIN_LOSSES, val_losses, val_map, best_map50, best_map50_95)

    return {
        'last_path': last_path,
        'best_path': best_path,
        'selected_epochs': selected_epochs,
        'best_epoch': best_epoch,
        'best_selection_value': best_selection_value,
        'best_map50': best_map50,
        'best_map50_95': best_map50_95,
    }


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        state = replay_epochs(save_dir)

        assert state['best_path'].is_file(), 'best.pt was not saved'
        assert state['last_path'].is_file(), 'last.pt was not saved'
        assert state['selected_epochs'] == [0, EXPECTED_BEST_EPOCH], (
            f'selected epochs {state["selected_epochs"]} != '
            f'{[0, EXPECTED_BEST_EPOCH]} (selection must follow mAP50:95)')

        best_ckpt = load_checkpoint(state['best_path'])
        last_ckpt = load_checkpoint(state['last_path'])

        print(f'selected epochs: {state["selected_epochs"]}')
        print(f'best.pt epoch: {best_ckpt["epoch"]}  '
              f'best_epoch: {best_ckpt["best_epoch"]}  '
              f'{best_ckpt["selection_metric"]}: '
              f'{best_ckpt["selection_value"]:.6f}')
        print(f'best.pt val_mAP50: {best_ckpt["val_map"]["mAP50"]:.6f}  '
              f'val_mAP50_95: {best_ckpt["val_map"]["mAP50_95"]:.6f}')
        print(f'last.pt epoch: {last_ckpt["epoch"]}')

        # --- required metadata on best.pt --------------------------------
        missing = REQUIRED_BEST_KEYS - set(best_ckpt)
        assert not missing, f'best.pt is missing keys: {sorted(missing)}'

        # --- best.pt is the mAP50:95-selected epoch ----------------------
        expected = EPOCHS[EXPECTED_BEST_EPOCH]
        assert best_ckpt['best_epoch'] == EXPECTED_BEST_EPOCH, (
            f'best_epoch {best_ckpt["best_epoch"]} != {EXPECTED_BEST_EPOCH}')
        assert best_ckpt['epoch'] == EXPECTED_BEST_EPOCH, (
            f'best.pt epoch {best_ckpt["epoch"]} != selected epoch '
            f'{EXPECTED_BEST_EPOCH}')
        assert best_ckpt['selection_metric'] == 'val_mAP50_95'
        assert best_ckpt['selection_value'] == \
            expected['val_map']['mAP50_95']
        assert best_ckpt['best_mAP50_95'] == \
            expected['val_map']['mAP50_95']
        # best_mAP50/best_mAP50_95 are running maxima as of the saved epoch.
        assert best_ckpt['best_mAP50'] == max(
            sample['val_map']['mAP50']
            for sample in EPOCHS[:EXPECTED_BEST_EPOCH + 1])

        # --- the audited pair: val_map/val_losses provenance -------------
        assert best_ckpt['val_map'] == expected['val_map'], (
            f'best.pt val_map {best_ckpt["val_map"]} is not the val_map of '
            f'epoch {EXPECTED_BEST_EPOCH} {expected["val_map"]}')
        assert best_ckpt['val_losses'] == expected['val_losses'], (
            'best.pt val_losses do not come from the selected epoch')

        # --- selection is not driven by val_total_loss -------------------
        loss_min_epoch = min(range(len(EPOCHS)), key=lambda i:
                             EPOCHS[i]['val_losses']['total_loss'])
        assert loss_min_epoch == EXPECTED_LOSS_MIN_EPOCH
        assert best_ckpt['best_epoch'] != loss_min_epoch, (
            'best.pt was chosen by val_total_loss')
        assert best_ckpt['val_map'] != \
            EPOCHS[loss_min_epoch]['val_map'], (
            'best.pt val_map matches the lowest-val_total_loss epoch')

        # --- KD metadata -------------------------------------------------
        assert best_ckpt['kd_layers'] == list(KD_LAYERS)
        assert best_ckpt['lambda_feat'] == LAMBDA_FEAT
        assert best_ckpt['teacher_weights'] == 'teacher.pt'
        assert best_ckpt['student_weights'] is None

        # --- last.pt always holds the final epoch ------------------------
        assert last_ckpt['epoch'] == len(EPOCHS) - 1, (
            f'last.pt epoch {last_ckpt["epoch"]} != final epoch '
            f'{len(EPOCHS) - 1}')
        assert last_ckpt['val_map'] == EPOCHS[-1]['val_map']
        assert last_ckpt['best_epoch'] == EXPECTED_BEST_EPOCH
        assert REQUIRED_BEST_KEYS <= set(last_ckpt)

    print('checkpoint audit passed: best.pt val_map comes from the '
          'mAP50:95-selected epoch, last.pt is the final epoch')


if __name__ == '__main__':
    main()
