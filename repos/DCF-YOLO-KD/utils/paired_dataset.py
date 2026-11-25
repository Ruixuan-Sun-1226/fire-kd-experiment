"""Paired RGB/IR dataset for cross-modal knowledge distillation.

Pairing contract
----------------
A pair is established *only* by identical file names, e.g.
``rgb_fire/images/train/3_frame_0000.jpg`` pairs with
``ir_fire/images/train/3_frame_0000.jpg``.  The two directories are indexed
into dictionaries and matched by key; we never ``zip`` two independently
sorted lists, so adding/removing a file cannot silently shift the pairing.

Augmentation contract
---------------------
Every sample returns the *same* geometric transform for RGB and IR:

* both images are letterboxed with one shared ratio/pad (their original
  sizes must match exactly, otherwise a clear error is raised);
* ``random_perspective`` is called twice with the *same* ``random`` state,
  which reproduces an identical affine matrix for both modalities;
* up/down and left/right flips are decided once per sample and applied to
  both images and both label sets.

Photometric augmentation (HSV) is applied to RGB only: it is not geometric,
and jittering the teacher's IR input would corrupt its targets.

Mosaic/mixup are intentionally not implemented here: mosaicking RGB and IR
independently would destroy the pairing.  This dataset is meant for paired
cross-modal training, not for the original single-modal recipe.

Output format
-------------
Each item is a dict with::

    rgb_image   uint8 CHW tensor (3, H, W), RGB order
    ir_image    uint8 CHW tensor (3, H, W), replicated to 3 channels if grayscale
    rgb_targets float tensor (N, 6): [image_index, class, x, y, w, h] normalized
    ir_targets  float tensor (M, 6): same format
    image_id    str, the shared file name (pairing key)

``image_index`` is filled in by :func:`collate_paired` so that the format
matches ``utils.loss.ComputeLoss`` expectations.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.datasets import augment_hsv, letterbox, random_perspective, xywhn2xyxy
from utils.general import xyxy2xywh

IMG_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')


def _list_images(directory, kind):
    """Return {file_name: path} for all images in ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'{kind} image directory does not exist: {directory}')
    index = {
        p.name: p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower().lstrip('.') in IMG_FORMATS
    }
    if not index:
        raise FileNotFoundError(f'No images found in {kind} directory: {directory}')
    return index


def _load_normalized_labels(label_path, single_cls=True):
    """Load YOLO labels as float32 (N, 5): [class, x, y, w, h] normalized."""
    label_path = Path(label_path)
    if not label_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append([float(v) for v in parts[:5]])
    labels = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
    if single_cls and labels.size:
        labels[:, 0] = 0  # everything is class 'fire'
    return labels


def _ensure_3ch(img, source):
    """Convert an image to 3 channels.

    IR images are often stored as single-channel grayscale.  The YOLO stem
    expects 3 input channels, so a single-channel IR image is replicated
    across all three channels (H, W) -> (H, W, 3).  RGB images are only
    normalized to 3 channels and otherwise left untouched.
    """
    if img is None:
        raise FileNotFoundError(f'Could not read image: {source}')
    if img.ndim == 2:  # grayscale / single-channel IR
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 4:  # BGRA -> BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


class PairedRGBIRDataset(Dataset):
    """RGB/IR fire-detection dataset paired by identical file name."""

    def __init__(self, rgb_root, ir_root, split='train', imgsz=640, augment=False,
                 hyp=None, stride=32, single_cls=True):
        self.rgb_root = Path(rgb_root)
        self.ir_root = Path(ir_root)
        self.rgb_img_dir = self.rgb_root / 'images' / split
        self.ir_img_dir = self.ir_root / 'images' / split
        self.rgb_label_dir = self.rgb_root / 'labels' / split
        self.ir_label_dir = self.ir_root / 'labels' / split

        imgsz = int(imgsz)
        stride = int(stride)
        if imgsz % stride != 0:
            raise ValueError(f'imgsz={imgsz} must be a multiple of stride={stride}')
        self.imgsz = imgsz
        self.stride = stride
        self.augment = augment
        self.hyp = hyp or {}
        self.single_cls = single_cls

        rgb_index = _list_images(self.rgb_img_dir, 'RGB')
        ir_index = _list_images(self.ir_img_dir, 'IR')

        # Pair strictly by file name.  ``sorted`` below only makes the epoch
        # order deterministic; it plays no role in deciding which files pair.
        rgb_names, ir_names = set(rgb_index), set(ir_index)
        if rgb_names != ir_names:
            missing_ir = sorted(rgb_names - ir_names)
            missing_rgb = sorted(ir_names - rgb_names)
            raise FileNotFoundError(
                'RGB/IR file names do not match.\n'
                f'  RGB without IR ({len(missing_ir)}): {missing_ir[:5]}\n'
                f'  IR without RGB ({len(missing_rgb)}): {missing_rgb[:5]}'
            )
        self.pairs = [(name, rgb_index[name], ir_index[name]) for name in sorted(rgb_names)]

        # Fail early if a label file is missing instead of silently training
        # on a background image.
        missing_labels = []
        for name, _, _ in self.pairs:
            stem = Path(name).stem
            if not (self.rgb_label_dir / f'{stem}.txt').is_file():
                missing_labels.append(f'rgb:{stem}.txt')
            if not (self.ir_label_dir / f'{stem}.txt').is_file():
                missing_labels.append(f'ir:{stem}.txt')
        if missing_labels:
            raise FileNotFoundError(
                f'Missing label files ({len(missing_labels)}): {missing_labels[:5]}'
            )

    def __len__(self):
        return len(self.pairs)

    def _labels_to_pixels(self, labels, ratio, pad, h0, w0):
        """Normalized xywh -> pixel xyxy in the letterboxed image."""
        labels = labels.copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:], ratio[0] * w0, ratio[1] * h0,
                padw=pad[0], padh=pad[1],
            )
        return labels

    @staticmethod
    def _normalize_labels(labels, img):
        n = len(labels)
        if n:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            labels[:, [2, 4]] /= img.shape[0]
            labels[:, [1, 3]] /= img.shape[1]
        return labels

    @staticmethod
    def _to_tensor(img):
        # BGR -> RGB and HWC -> CHW, matching utils.datasets.LoadImagesAndLabels
        return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))

    @staticmethod
    def _to_target(labels):
        out = torch.zeros((len(labels), 6))
        if len(labels):
            out[:, 1:] = torch.from_numpy(labels)
        return out

    def __getitem__(self, index):
        name, rgb_path, ir_path = self.pairs[index]

        rgb = _ensure_3ch(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), rgb_path)
        # IMREAD_UNCHANGED keeps a grayscale IR file as HxW so _ensure_3ch
        # can explicitly replicate it to 3 channels.
        ir = _ensure_3ch(cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED), ir_path)

        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(
                f'Paired images differ in size for "{name}": '
                f'RGB {rgb.shape[:2]} vs IR {ir.shape[:2]}'
            )
        h0, w0 = rgb.shape[:2]

        # One geometric letterbox for both modalities.
        rgb, ratio, pad = letterbox(rgb, self.imgsz, auto=False, scaleup=self.augment)
        ir, ratio_ir, pad_ir = letterbox(ir, self.imgsz, auto=False, scaleup=self.augment)
        if ratio != ratio_ir or pad != pad_ir:
            raise RuntimeError(f'Letterbox mismatch for pair "{name}"')

        rgb_labels = _load_normalized_labels(
            self.rgb_label_dir / f'{Path(name).stem}.txt', self.single_cls)
        ir_labels = _load_normalized_labels(
            self.ir_label_dir / f'{Path(name).stem}.txt', self.single_cls)
        rgb_labels = self._labels_to_pixels(rgb_labels, ratio, pad, h0, w0)
        ir_labels = self._labels_to_pixels(ir_labels, ratio, pad, h0, w0)

        if self.augment:
            h = self.hyp
            geom_kwargs = dict(
                degrees=h.get('degrees', 0.0),
                translate=h.get('translate', 0.1),
                scale=h.get('scale', 0.5),
                shear=h.get('shear', 0.0),
                perspective=h.get('perspective', 0.0),
            )
            # Replay the exact same random stream for IR so both modalities
            # receive an identical affine warp.  random_perspective draws all
            # of its parameters before touching the pixels/labels.
            rng_state = random.getstate()
            rgb, rgb_labels = random_perspective(rgb, rgb_labels, **geom_kwargs)
            random.setstate(rng_state)
            ir, ir_labels = random_perspective(ir, ir_labels, **geom_kwargs)

            # Photometric jitter is RGB-only (not a geometric transform).
            augment_hsv(rgb, hgain=h.get('hsv_h', 0.015),
                        sgain=h.get('hsv_s', 0.7), vgain=h.get('hsv_v', 0.4))

        rgb_labels = self._normalize_labels(rgb_labels, rgb)
        ir_labels = self._normalize_labels(ir_labels, ir)

        if self.augment:
            h = self.hyp
            # Draw each flip decision once and apply it to both modalities.
            flipud = random.random() < h.get('flipud', 0.0)
            fliplr = random.random() < h.get('fliplr', 0.5)
            if flipud:
                rgb, ir = np.flipud(rgb), np.flipud(ir)
                for labels in (rgb_labels, ir_labels):
                    if len(labels):
                        labels[:, 2] = 1 - labels[:, 2]
            if fliplr:
                rgb, ir = np.fliplr(rgb), np.fliplr(ir)
                for labels in (rgb_labels, ir_labels):
                    if len(labels):
                        labels[:, 1] = 1 - labels[:, 1]

        return {
            'rgb_image': self._to_tensor(rgb),
            'ir_image': self._to_tensor(ir),
            'rgb_targets': self._to_target(rgb_labels),
            'ir_targets': self._to_target(ir_labels),
            'image_id': name,
            'rgb_path': str(rgb_path),
            'ir_path': str(ir_path),
        }


def collate_paired(batch):
    """Stack images and concatenate targets, filling target column 0."""
    rgb_images = torch.stack([s['rgb_image'] for s in batch], 0)
    ir_images = torch.stack([s['ir_image'] for s in batch], 0)

    def cat_targets(key):
        parts = []
        for image_index, sample in enumerate(batch):
            targets = sample[key].clone()
            if targets.numel():
                targets[:, 0] = image_index
                parts.append(targets)
        return torch.cat(parts, 0) if parts else torch.zeros((0, 6))

    return {
        'rgb_image': rgb_images,
        'ir_image': ir_images,
        'rgb_targets': cat_targets('rgb_targets'),
        'ir_targets': cat_targets('ir_targets'),
        'image_id': [s['image_id'] for s in batch],
    }
