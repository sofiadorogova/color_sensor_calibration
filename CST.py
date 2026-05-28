from dataclasses import dataclass

import numpy as np
from constants import CST_MODELS


@dataclass
class TransformFeatures:
    model_name: str

    def PCC2(self, rgb: np.ndarray):
        return np.concatenate(
            [
                rgb,
                rgb**2,
                rgb[..., :1] * rgb[..., 1:2],
                rgb[..., 1:2] * rgb[..., 2:],
                rgb[..., :1] * rgb[..., 2:],
            ],
            axis=1,
        )

    def PCC3(self, rgb: np.ndarray):
        return np.concatenate(
            [
                rgb,
                rgb**2,
                rgb[..., :1] * rgb[..., 1:2],
                rgb[..., 1:2] * rgb[..., 2:],
                rgb[..., :1] * rgb[..., 2:],
                rgb**3,
                rgb[..., :1] ** 2 * rgb[..., 1:2],
                rgb[..., :1] ** 2 * rgb[..., 2:],
                rgb[..., 1:2] ** 2 * rgb[..., :1],
                rgb[..., 1:2] ** 2 * rgb[..., 2:],
                rgb[..., 2:] ** 2 * rgb[..., :1],
                rgb[..., 2:] ** 2 * rgb[..., 1:2],
                rgb[..., :1] * rgb[..., 1:2] * rgb[..., 2:],
            ],
            axis=1,
        )

    def RPCC2(self, rgb: np.ndarray):
        return np.concatenate(
            [
                rgb,
                (rgb[:, :1] * rgb[:, 1:2]) ** (1 / 2),
                (rgb[:, 1:2] * rgb[:, 2:]) ** (1 / 2),
                (rgb[:, :1] * rgb[:, 2:]) ** (1 / 2),
            ],
            axis=1,
        )

    def RPCC3(self, rgb: np.ndarray):
        return np.concatenate(
            [
                rgb,
                (rgb[..., :1] * rgb[..., 1:2]) ** (1 / 2),
                (rgb[..., 1:2] * rgb[..., 2:]) ** (1 / 2),
                (rgb[..., :1] * rgb[..., 2:]) ** (1 / 2),
                (rgb[..., :1] ** 2 * rgb[..., 1:2]) ** (1 / 3),
                (rgb[..., :1] ** 2 * rgb[..., 2:]) ** (1 / 3),
                (rgb[..., 1:2] ** 2 * rgb[..., :1]) ** (1 / 3),
                (rgb[..., 1:2] ** 2 * rgb[..., 2:]) ** (1 / 3),
                (rgb[..., 2:] ** 2 * rgb[..., :1]) ** (1 / 3),
                (rgb[..., 2:] ** 2 * rgb[..., 1:2]) ** (1 / 3),
                (rgb[..., :1] * rgb[..., 1:2] * rgb[..., 2:]) ** (1 / 3),
            ],
            axis=1,
        )

    def __call__(self, rgb: np.ndarray):
        methods = {
            "PCC2": self.PCC2,
            "PCC3": self.PCC3,
            "RPCC2": self.RPCC2,
            "RPCC3": self.RPCC3,
        }
        return methods[self.model_name](rgb)


def apply_CST(rgb: np.ndarray, CST_matrix: np.ndarray) -> np.ndarray:
    features_num = CST_matrix.T.shape[-1]
    match features_num:
        case 9:
            method = "PCC2"
        case 19:
            method = "PCC3"
        case 6:
            method = "RPCC2"
        case 13:
            method = "RPCC3"
        case _:
            raise ValueError("Only this methods are supported.")
    return TransformFeatures(method)(rgb) @ CST_matrix


def get_CST(rgb_src: np.ndarray, rgb_dst: np.ndarray, model_name: str) -> np.ndarray:
    assert model_name in CST_MODELS
    rgb_features = TransformFeatures(model_name)(rgb_src)
    return fit(rgb_features, rgb_dst)


def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]
