import numpy as np
from scipy.stats import pearsonr


def calc_angle(x: np.ndarray, y: np.ndarray):
    products = (x * y).sum(axis=-1)
    magnitudes = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-8
    cos_theta = products / (magnitudes + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta)).mean()


def calc_pearson(x: np.ndarray, y: np.ndarray):
    return pearsonr(x.flatten(), y.flatten()).statistic


def compute_metrics(x: np.ndarray, y: np.ndarray):
    pearson = calc_pearson(x, y)
    angle = calc_angle(x, y)
    return angle, pearson
