import csv
from collections import Counter
from pathlib import Path


RAW_PATH = Path("data/raw/iot_telemetry_data.csv")


def to_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}\n"
            "Place iot_telemetry_data.csv inside data/raw/ first."
        )

    numeric_cols = ["co", "humidity", "lpg", "smoke", "temp"]
    bool_cols = ["light", "motion"]

    row_count = 0
    devices = Counter()
    bool_counts = {col: Counter() for col in bool_cols}
    numeric_stats = {
        col: {"count": 0, "min": None, "max": None, "sum": 0.0}
        for col in numeric_cols
    }

    with RAW_PATH.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        reader = csv.DictReader(fp)
        columns = reader.fieldnames

        for row in reader:
            row_count += 1
            devices[row["device"]] += 1

            for col in bool_cols:
                bool_counts[col][row[col]] += 1

            for col in numeric_cols:
                value = to_float(row[col])
                if value is None:
                    continue

                stats = numeric_stats[col]
                stats["count"] += 1
                stats["sum"] += value
                stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                stats["max"] = value if stats["max"] is None else max(stats["max"], value)

    print("Dataset:", RAW_PATH)
    print("Columns:", columns)
    print("Rows:", row_count)

    print("\nDevices:")
    for device, count in devices.most_common():
        print(f"  {device}: {count}")

    print("\nBoolean columns:")
    for col, counts in bool_counts.items():
        print(f"  {col}: {dict(counts)}")

    print("\nNumeric columns:")
    for col, stats in numeric_stats.items():
        mean = stats["sum"] / stats["count"] if stats["count"] else None
        print(
            f"  {col}: "
            f"count={stats['count']}, "
            f"min={stats['min']:.4f}, "
            f"max={stats['max']:.4f}, "
            f"mean={mean:.4f}"
        )


if __name__ == "__main__":
    main()
