import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import CST_MODELS, Paths
from CST import apply_CST, get_CST
from dataset_processing import Lin2sRGB, create_dataset, sRGB2Lin
from metrics import compute_metrics
from PIL import Image
from tqdm import tqdm


def get_matched_pairs(
    config_path: str = "dataset_config.json",
) -> list[tuple[int, int]]:
    """Загружает проверенные пары изображений из конфигурационного файла."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    pairs = [tuple(pair) for pair in config["matched_pairs"]]
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Калибровка цветового сенсора")
    parser.add_argument("-o", "--output", type=Path, default="experiment_results")
    parser.add_argument("-n", "--train_size", type=float, default=0.8)
    parser.add_argument("--src_cam", default="Canon1DsMkIII")
    parser.add_argument("--dst_cam", default="Canon600D")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_test_split_pairs(pairs: list[tuple[int, int]], train_size: float = 0.8):
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_size)
    return shuffled[:split_idx][0], shuffled[split_idx:][0]


def eval_model(data: tuple, cst_matrix: np.ndarray):
    src_gt, dst_gt = data
    dst_pred = apply_CST(src_gt, cst_matrix)

    dst_pred_angle, dst_pred_pearson = compute_metrics(dst_pred, dst_gt)
    src_dst_angle, src_dst_pearson = compute_metrics(src_gt[:, :3], dst_gt)

    return dst_pred_angle, dst_pred_pearson, src_dst_angle, src_dst_pearson


def visual_results(
    src_idx: int, dst_idx: int, output_dir: Path, src_cam: str, dst_cam: str
):
    def smart_roi_crop(img_path: str, mask_path: str, padding: int = 150) -> np.ndarray:
        img = np.array(Image.open(img_path).convert("RGB"))

        with open(mask_path, "r") as f:
            lines = f.readlines()

        roi = np.array([float(i) for i in lines[0].strip().split(",")]) * 2.0
        x, y, w, h = [int(v) for v in roi]

        y_start = max(0, y - padding)
        y_end = min(img.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(img.shape[1], x + w + padding)

        if y_start >= y_end or x_start >= x_end:
            print(f"Предупреждение: Некорректная маска у {Path(img_path).name}.")
            return img

        crop = img[y_start:y_end, x_start:x_end]

        y_patch_1 = np.mean([float(j) for j in lines[2].strip().split(",")]) * 2.0 + y
        y_patch_24 = np.mean([float(j) for j in lines[48].strip().split(",")]) * 2.0 + y

        if y_patch_1 > y_patch_24:
            # Проверка ориентации мишени (если 1-й патч ниже 24-го, разворачиваем)
            crop = crop[::-1, ::-1, :]

        return crop

    src_img_path = Paths.imgs(src_cam, src_idx)
    src_mask_path = Paths.markups(src_cam, src_idx)
    dst_img_path = Paths.imgs(dst_cam, dst_idx)
    dst_mask_path = Paths.markups(dst_cam, dst_idx)

    src = smart_roi_crop(src_img_path, src_mask_path)
    gt_dst = smart_roi_crop(dst_img_path, dst_mask_path)

    h, w, c = src.shape

    if gt_dst.shape != src.shape:
        gt_img = Image.fromarray(gt_dst).resize((w, h))
        gt_dst = np.array(gt_img)

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(24, 16))

    axs[0, 0].imshow(src)
    axs[0, 0].set_title(f"Исходное изображение ({src_cam}_{src_idx:04d})", fontsize=18)
    axs[0, 1].imshow(gt_dst)
    axs[0, 1].set_title(f"Эталон / Target ({dst_cam}_{dst_idx:04d})", fontsize=18)

    lin_src = sRGB2Lin(np.array(src).reshape(-1, 3))
    lin_dst = sRGB2Lin(np.array(gt_dst).reshape(-1, 3))

    for k, model_name in enumerate(CST_MODELS):
        i, j = (k + 2) // ncols, (k + 2) % ncols
        model_matrix = np.load(output_dir / f"{model_name}.npy")

        lin_pred_vector = apply_CST(lin_src, model_matrix)
        srgb_pred_img = Lin2sRGB(lin_pred_vector).reshape(h, w, 3)

        axs[i, j].imshow(srgb_pred_img)
        a, p = compute_metrics(lin_pred_vector, lin_dst)
        axs[i, j].set_title(f"Модель: {model_name}\n Пирсон = {p:.3f}", fontsize=16)

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(
        output_dir / f"visual_match_{src_idx:04d}_{dst_idx:04d}.png",
        bbox_inches="tight",
    )
    plt.close()


def numerical_results(data: list, output_dir: Path, test_size: int):
    """Построение и сохранение графиков с результатами метрик."""
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    df = df.rename(
        columns={
            "gt-pred": "После калибровки",
            "gt-src": "Базовая ошибка (до калибровки)",
        },
        level=0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    fig.suptitle(
        f"Сравнение алгоритмов цветовой калибровки сенсоров\n(Метрики усреднены по {test_size} тестовым изображениям)",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    angles = df.xs("angle", level=1, axis=1)
    angles.plot.bar(ax=ax1, rot=0, width=0.7)
    ax1.set_title("Угловая ошибка (Angular Error)\nМеньше — лучше", fontsize=14)
    ax1.set_xlabel("Алгоритм", fontsize=12)
    ax1.set_ylabel("Ошибка (градусы)", fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    pearsons = df.xs("pearson", level=1, axis=1)
    pearsons.plot.bar(ax=ax2, rot=0, width=0.7)
    ax2.set_title("Коэффициент корреляции Пирсона\nБольше — лучше", fontsize=14)
    ax2.set_xlabel("Алгоритм", fontsize=12)
    ax2.set_ylabel("Значение корреляции", fontsize=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    for ax in (ax1, ax2):
        ax.set_xticklabels(df[("Model", "")], rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "numerical.png", bbox_inches="tight", dpi=150)
    plt.close()

    df.to_csv(output_dir / "results.csv")


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    all_pairs = get_matched_pairs("dataset_config.json")

    single_pair = [all_pairs[1]]
    print(f"Выбрана одна пара для эксперимента: {single_pair}")

    train_pairs = single_pair
    test_pairs = single_pair

    print(f"Обучающая выборка: {len(train_pairs)} пар")
    print(f"Тестовая выборка: {len(test_pairs)} пар")

    train_src, train_dst = create_dataset(train_pairs, args.src_cam, args.dst_cam)
    test_src, test_dst = create_dataset(test_pairs, args.src_cam, args.dst_cam)

    experiment_folder: Path = args.output
    experiment_folder.mkdir(exist_ok=True)

    data = []
    for model_name in tqdm(CST_MODELS, desc="Обучение и оценка моделей"):
        matrix = get_CST(train_src, train_dst, model_name)
        np.save(experiment_folder / f"{model_name}.npy", matrix)

        # Тестируем на той же самой картинке
        angle1, pearson1, angle2, pearson2 = eval_model((test_src, test_dst), matrix)
        data.append(
            {
                ("Model", ""): model_name,
                ("gt-pred", "angle"): angle1,
                ("gt-pred", "pearson"): pearson1,
                ("gt-src", "angle"): angle2,
                ("gt-src", "pearson"): pearson2,
            }
        )

    numerical_results(data, experiment_folder, len(test_pairs))

    # Визуализируем эту же пару
    sample_src_idx, sample_dst_idx = test_pairs[0]
    print(
        f"Генерация визуализации для тестовой пары: {sample_src_idx:04d} -> {sample_dst_idx:04d} ..."
    )
    visual_results(
        sample_src_idx, sample_dst_idx, experiment_folder, args.src_cam, args.dst_cam
    )

    print(
        f"Вычисление завершено. Результаты сохранены в директорию: {experiment_folder}"
    )
