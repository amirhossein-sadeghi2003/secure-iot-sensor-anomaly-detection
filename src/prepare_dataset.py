from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/iot_telemetry_data.csv")
OUTPUT_PATH = Path("data/processed/clean_iot_telemetry.csv")


def bool_to_int(value):
    value = str(value).strip().lower()

    if value == "true":
        return 1
    if value == "false":
        return 0

    return pd.NA


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}\n"
            "Download the Kaggle IoT telemetry dataset and place "
            "iot_telemetry_data.csv inside data/raw/."
        )

    df = pd.read_csv(RAW_PATH)

    df = df.rename(columns={
        "ts": "timestamp",
        "co": "co_level",
        "lpg": "lpg_level",
        "smoke": "smoke_level",
        "temp": "temperature_c",
    })

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    for col in ["co_level", "humidity", "lpg_level", "smoke_level", "temperature_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["light"] = df["light"].apply(bool_to_int)
    df["motion"] = df["motion"].apply(bool_to_int)

    df = df.dropna(subset=[
        "timestamp",
        "datetime",
        "device",
        "co_level",
        "humidity",
        "light",
        "lpg_level",
        "motion",
        "smoke_level",
        "temperature_c",
    ])

    df = df.sort_values(["device", "timestamp"]).reset_index(drop=True)

    df["device_time_step"] = df.groupby("device").cumcount()
    df["elapsed_seconds"] = df.groupby("device")["timestamp"].transform(
        lambda values: values - values.iloc[0]
    )

    output_columns = [
        "timestamp",
        "datetime",
        "elapsed_seconds",
        "device_time_step",
        "device",
        "co_level",
        "humidity",
        "light",
        "lpg_level",
        "motion",
        "smoke_level",
        "temperature_c",
    ]

    cleaned = df[output_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print(f"Rows: {len(cleaned)}")
    print(f"Devices: {cleaned['device'].nunique()}")
    print()
    print("Rows per device:")
    print(cleaned["device"].value_counts())
    print()
    print("Preview:")
    print(cleaned.head())


if __name__ == "__main__":
    main()
