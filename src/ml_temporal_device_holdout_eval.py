import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ml_temporal_detector import SENSOR_COLUMNS, add_temporal_features


INPUT_PATH = "data/processed/labeled_iot_spoofing_data.csv"
RESULTS_DIR = "results"
METRICS_PATH = os.path.join(RESULTS_DIR, "ml_temporal_device_holdout_metrics.txt")
CSV_PATH = os.path.join(RESULTS_DIR, "ml_temporal_device_holdout_results.csv")


def evaluate_held_out_device(feature_df, feature_columns, held_out_device):
    train_df = feature_df[feature_df["device"] != held_out_device].copy()
    test_df = feature_df[feature_df["device"] == held_out_device].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["attack_label"].astype(int)

    X_test = test_df[feature_columns]
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

    row = {
        "held_out_device": held_out_device,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    attack_result = test_df[["attack_type"]].copy()
    attack_result["y_true"] = y_test.to_numpy()
    attack_result["y_pred"] = y_pred

    attack_rows = attack_result[attack_result["y_true"] == 1]
    attack_rates = attack_rows.groupby("attack_type")["y_pred"].mean().sort_values(ascending=False)

    return row, attack_rates, attack_rows["attack_type"].value_counts()


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

    devices = sorted(feature_df["device"].unique())

    rows = []
    report_lines = []

    report_lines.append("Temporal ML detector with device-held-out evaluation")
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Rows: {len(df)}")
    report_lines.append("Split strategy: train on two devices, test on one held-out device")
    report_lines.append("")
    report_lines.append("Model:")
    report_lines.append("RandomForestClassifier with snapshot and temporal features")
    report_lines.append("")

    for device in devices:
        row, attack_rates, attack_counts = evaluate_held_out_device(
            feature_df=feature_df,
            feature_columns=feature_columns,
            held_out_device=device,
        )

        rows.append(row)

        report_lines.append("=" * 60)
        report_lines.append(f"Held-out device: {device}")
        report_lines.append(f"Training samples: {row['train_samples']}")
        report_lines.append(f"Test samples: {row['test_samples']}")
        report_lines.append("")
        report_lines.append("Confusion matrix [normal, attack]:")
        report_lines.append(f"[[{row['tn']} {row['fp']}]")
        report_lines.append(f" [{row['fn']} {row['tp']}]]")
        report_lines.append("")
        report_lines.append(f"Accuracy:  {row['accuracy']:.4f}")
        report_lines.append(f"Precision: {row['precision']:.4f}")
        report_lines.append(f"Recall:    {row['recall']:.4f}")
        report_lines.append(f"F1-score:  {row['f1']:.4f}")
        report_lines.append("")
        report_lines.append("Detection rate by attack type:")
        for attack_type, rate in attack_rates.items():
            count = int(attack_counts[attack_type])
            report_lines.append(f"- {attack_type}: {rate:.4f} ({count} samples)")
        report_lines.append("")

    results_df = pd.DataFrame(rows)

    report_lines.append("=" * 60)
    report_lines.append("Average across held-out devices:")
    report_lines.append(f"Accuracy:  {results_df['accuracy'].mean():.4f}")
    report_lines.append(f"Precision: {results_df['precision'].mean():.4f}")
    report_lines.append(f"Recall:    {results_df['recall'].mean():.4f}")
    report_lines.append(f"F1-score:  {results_df['f1'].mean():.4f}")

    output = "\n".join(report_lines)
    print(output)

    results_df.to_csv(CSV_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print("")
    print(f"Saved CSV results to: {CSV_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
