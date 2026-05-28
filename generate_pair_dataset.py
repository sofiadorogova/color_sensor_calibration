import json
from pathlib import Path


def generate_json_config():
    pairs = []

    # Диапазон 1 (со сдвигом): Canon 1Ds 0239-0259 -> Canon 600D 0004-0024
    for s, t in zip(range(239, 260), range(4, 25)):
        if s in [
            258,
            255,
            248,
            245,
        ]:  # Исключаем пары с неверной разметкой
            continue
        pairs.append([s, t])

    # Диапазон 2: Canon 1Ds 0183-0221 -> Canon 600D 0099-0137
    for s, t in zip(range(183, 222), range(99, 138)):
        if s in [220, 200]:  # Исключаем пары с неверной разметкой
            continue
        pairs.append([s, t])

    config_data = {
        "cameras": {"source": "Canon1DsMkIII", "target": "Canon600D"},
        "matched_pairs": pairs,
    }

    config_path = Path("dataset_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print(f"Обновленный конфиг успешно сохранен в {config_path.resolve()}")
    print(f"Всего валидных пар сохранено: {len(pairs)}")


if __name__ == "__main__":
    generate_json_config()
