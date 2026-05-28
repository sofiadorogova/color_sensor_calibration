from pathlib import Path

import numpy as np
from constants import Paths
from PIL import Image
from tqdm import tqdm


def sRGB2Lin(srgb: np.ndarray) -> np.ndarray:
    srgb_norm = (srgb / 255.0).astype(np.float32)
    linrgb = np.where(
        srgb_norm > 0.04045, ((srgb_norm + 0.055) / 1.055) ** 2.4, srgb_norm / 12.92
    )
    return linrgb


def Lin2sRGB(linrgb: np.ndarray) -> np.ndarray:
    linrgb = np.clip(linrgb, 0.0, 1.0)
    srgb = np.where(
        linrgb > 0.0031308, 1.055 * (linrgb ** (1 / 2.4)) - 0.055, linrgb * 12.92
    )
    return (srgb * 255).astype(np.uint8)


def crop_patches(
    img_idx: int, camera_name: str, margin_ratio: float = 0.25
) -> list[np.ndarray]:
    img_path = Paths.imgs(camera_name, img_idx)
    mask_path = Paths.markups(camera_name, img_idx)

    if not Path(img_path).exists() or not Path(mask_path).exists():
        return []

    img_arr = np.array(Image.open(img_path).convert("RGB"))
    img_h, img_w = img_arr.shape[:2]

    with open(mask_path) as f:
        lines = f.readlines()

    scale = 2.0
    roi = np.array([float(i) for i in lines[0].strip().split(",")]) * scale
    roi_x, roi_y = roi[0], roi[1]

    patches = []
    for i in range(1, 48, 2):
        x_coords = (
            np.array([float(j) for j in lines[i].strip().split(",")]) * scale
        ) + roi_x
        y_coords = (
            np.array([float(j) for j in lines[i + 1].strip().split(",")]) * scale
        ) + roi_y

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        w = x_max - x_min
        h = y_max - y_min

        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x_min += margin_x
        x_max -= margin_x
        y_min += margin_y
        y_max -= margin_y

        x_min, x_max = max(0, x_min), min(img_w, x_max)
        y_min, y_max = max(0, y_min), min(img_h, y_max)

        if x_min >= x_max or y_min >= y_max:
            return []

        patch_crop = img_arr[y_min:y_max, x_min:x_max]
        patches.append(patch_crop)

    return patches


def avg_color_per_patch(patches: list[np.ndarray]) -> np.ndarray:
    # сначала линеаризация (sRGB2Lin), потом усреднение
    return np.stack([np.mean(patch, axis=(0, 1)) for patch in patches], axis=0)


def create_dataset(
    pairs: list[tuple[int, int]], src_cam: str, dst_cam: str
) -> tuple[np.ndarray, np.ndarray]:
    src_patches, dst_patches = [], []

    for src_num, dst_num in tqdm(pairs, desc="Creating Dataset", leave=False):
        src_crops = crop_patches(src_num, src_cam, margin_ratio=0.15)
        dst_crops = crop_patches(dst_num, dst_cam, margin_ratio=0.15)

        if len(src_crops) == 24 and len(dst_crops) == 24:
            src_lin = sRGB2Lin(avg_color_per_patch(src_crops))
            dst_lin = sRGB2Lin(avg_color_per_patch(dst_crops))
            src_patches.append(src_lin)
            dst_patches.append(dst_lin)

    if not src_patches:
        raise ValueError("Датасет пуст. Проверить пути.")

    return np.concatenate(src_patches, axis=0), np.concatenate(dst_patches, axis=0)
