from dataclasses import dataclass
from pathlib import Path

CAMERA_NAMES = ["Canon1DsMkIII", "Canon600D"]
CST_MODELS: list[str] = ["PCC2", "PCC3", "RPCC2", "RPCC3"]


@dataclass
class PathFormatter:
    paths: dict

    def __getitem__(self, key: str):
        return self.paths[key]

    def __call__(self, key: str, idx: int):
        base_name = self.paths[key] / f"{key}_{idx:04}"
        if "JPG" in str(base_name):
            return f"{base_name}.jpg"
        return f"{base_name}_mask.txt"


@dataclass
class Paths:
    dataset: Path = Path(
        "/beta/users/dorogova/color_calibration/color_sensor_calibration/data"
    )

    markups: PathFormatter = PathFormatter(
        {
            "Canon1DsMkIII": dataset / "Canon1DsMkIII_CHECKER/CHECKER",
            "Canon600D": dataset / "Canon600D_CHECKER/CHECKER",
        }
    )
    imgs: PathFormatter = PathFormatter(
        {
            "Canon1DsMkIII": dataset / "Canon1DsMkIII_JPEG/JPG",
            "Canon600D": dataset / "Canon600D_JPEG/JPG",
        }
    )
