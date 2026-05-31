import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml_temporal_detector import SENSOR_COLUMNS, add_temporal_features


INPUT_PATH = "data/processed/labeled_iot_spoofing_data.csv"
RESULTS_DIR = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "ml_temporal_time_split_metrics.txt")


def make_time_split(feature_df, train_fraction=0.70):
    train_parts = []
    test_parts = []

    ordered = feature_df.sort_values(["device", "elapsed_seconds"]).copy()

    for _, part in ordered.groupby("device"):
        split_index = int(len(part) * train_fraction)

        if split_index <= 0 or split_index >= len(part):
            raise ValueError("Invalid split for one of the devices.")

        train_parts.append(part.iloc[:split_index])
        test_parts.append(part.iloc[split_index:])

    train_df = pd.concat(train_parts).sort_index()
    test_df = pd.concat(test_parts).sort_index()

    return train_df, test_df


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required = ["device", "elapsed_seconds", "attack_label", "attack_type"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_sensor_cols = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing_sensor_cols:
        raise ValueError(f"Missing sensor columns: {missing_sensor_cols}")

    feature_df, temporal_cols = add_temporal_features(df)
    feature_columns = SENSOR_COLUMNS + temporal_cols

    train_df, test_df = make_time_split(feature_df, train_fraction=0.70)

    X_train = train_df[feature_columns].copy()
    y_train = train_df["attack_label"].astype(int)

    X_test = test_df[feature_columns].copy()
    y_test = test_df["attack_label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    lines = []
    lines.append("Temporal ML detector with device-wise time-based split")
    lines.append("")
    lines.append(f"Input file: {INPUT_PATH}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Training samples: {len(train_df)}")
    lines.append(f"Test samples: {len(test_df)}")
    lines.append("Split strategy: first 70% of each device timeline for training, last 30% for testing")
    lines.append("")
    lines.append("Model:")
    lines.append("RandomForestClassifier with snapshot and temporal features")
    lines.append("")
    lines.append("Confusion matrix [normal, attack]:")
    lines.append(str(cm))
    lines.append("")
    lines.append(f"Accuracy:  {acc:.4f}")
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall:    {recall:.4f}")
    lines.append(f"F1-score:  {f1:.4f}")

    result = test_df[["device", "elapsed_seconds", "attack_type"]].copy()
    result["y_true"] = y_test.to_numpy()
    result["y_pred"] = y_pred

    attack_rows = result[result["y_true"] == 1]
    rates = attack_rows.groupby("attack_type")["y_pred"].mean().sort_values(ascending=False)

    lines.append("")
    lines.append("Detection rate by attack type on time-based test set:")
    for attack_type, rate in rates.items():
        count = int((attack_rows["attack_type"] == attack_type).sum())
        lines.append(f"- {attack_type}: {rate:.4f} ({count} samples)")

    lines.append("")
    lines.append("Test samples by device:")
    for device, count in test_df["device"].value_counts().items():
        lines.append(f"- {device}: {count}")

    output = "\n".join(lines)
    print(output)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print("")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
