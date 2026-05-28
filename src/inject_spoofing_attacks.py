from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/clean_iot_telemetry.csv")
OUTPUT_PATH = Path("data/processed/labeled_iot_spoofing_data.csv")

SEED = 42
NORMAL_SAMPLES = 30000
ATTACK_SAMPLES = 12000

SENSOR_COLUMNS = [
    "co_level",
    "humidity",
    "light",
    "lpg_level",
    "motion",
    "smoke_level",
    "temperature_c",
]


def require_input():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {INPUT_PATH}\n"
            "Run this first:\n"
            "  python src/prepare_dataset.py"
        )


def build_thresholds(df):
    return {
        "temp_low": df["temperature_c"].quantile(0.01) - 5,
        "temp_high": df["temperature_c"].quantile(0.99) + 8,
        "humidity_low": max(0, df["humidity"].quantile(0.01) - 20),
        "co_high": df["co_level"].quantile(0.99) * 1.8,
        "lpg_high": df["lpg_level"].quantile(0.99) * 1.8,
        "smoke_high": df["smoke_level"].quantile(0.99) * 1.8,
    }


def apply_temperature_spoof(row, thresholds, rng):
    row["temperature_c"] = rng.choice([
        thresholds["temp_low"],
        thresholds["temp_high"],
    ]) + rng.normal(0, 0.3)
    return row


def apply_gas_spoof(row, thresholds, rng):
    row["co_level"] = thresholds["co_high"] + rng.normal(0, 0.0002)
    row["lpg_level"] = thresholds["lpg_high"] + rng.normal(0, 0.0002)
    row["smoke_level"] = thresholds["smoke_high"] + rng.normal(0, 0.0005)
    return row


def apply_light_motion_mismatch(row, rng):
    hour = pd.to_datetime(row["datetime"]).hour

    if 7 <= hour <= 18:
        row["light"] = 0
    else:
        row["light"] = 1

    row["motion"] = 1
    return row


def apply_replay_like_values(row, replay_row):
    for col in SENSOR_COLUMNS:
        row[col] = replay_row[col]
    return row


def apply_mixed_false_data(row, thresholds, rng):
    row["temperature_c"] = thresholds["temp_high"] + rng.normal(0, 0.5)
    row["humidity"] = thresholds["humidity_low"] + rng.normal(0, 1.0)
    row["co_level"] = thresholds["co_high"] + rng.normal(0, 0.0002)
    row["lpg_level"] = thresholds["lpg_high"] + rng.normal(0, 0.0002)
    row["smoke_level"] = thresholds["smoke_high"] + rng.normal(0, 0.0005)
    row["light"] = 1
    row["motion"] = 0
    return row


def make_labeled_dataset(df):
    rng = np.random.default_rng(SEED)
    thresholds = build_thresholds(df)

    normal = df.sample(
        n=min(NORMAL_SAMPLES, len(df)),
        random_state=SEED,
    ).copy()
    normal["attack_label"] = 0
    normal["attack_type"] = "normal"

    attack_base = df.sample(
        n=ATTACK_SAMPLES,
        replace=True,
        random_state=SEED + 1,
    ).reset_index(drop=True)

    replay_rows = df.sample(
        n=ATTACK_SAMPLES,
        replace=True,
        random_state=SEED + 2,
    ).reset_index(drop=True)

    attack_types = rng.choice(
        [
            "temperature_spoof",
            "gas_spoof",
            "light_motion_mismatch",
            "replay_like_values",
            "mixed_false_data",
        ],
        size=ATTACK_SAMPLES,
    )

    attacked_rows = []

    for i, row in attack_base.iterrows():
        row = row.copy()
        attack_type = attack_types[i]

        if attack_type == "temperature_spoof":
            row = apply_temperature_spoof(row, thresholds, rng)

        elif attack_type == "gas_spoof":
            row = apply_gas_spoof(row, thresholds, rng)

        elif attack_type == "light_motion_mismatch":
            row = apply_light_motion_mismatch(row, rng)

        elif attack_type == "replay_like_values":
            row = apply_replay_like_values(row, replay_rows.iloc[i])

        elif attack_type == "mixed_false_data":
            row = apply_mixed_false_data(row, thresholds, rng)

        row["attack_label"] = 1
        row["attack_type"] = attack_type
        attacked_rows.append(row)

    attacks = pd.DataFrame(attacked_rows)

    labeled = pd.concat([normal, attacks], ignore_index=True)
    labeled = labeled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return labeled


def main():
    require_input()

    df = pd.read_csv(INPUT_PATH)
    labeled = make_labeled_dataset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved labeled dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(labeled)}")
    print()
    print("Attack label counts:")
    print(labeled["attack_label"].value_counts().sort_index())
    print()
    print("Attack type counts:")
    print(labeled["attack_type"].value_counts())
    print()
    print("Preview:")
    print(labeled.head())


if __name__ == "__main__":
    main()
