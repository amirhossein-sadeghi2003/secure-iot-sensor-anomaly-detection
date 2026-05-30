import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/labeled_iot_spoofing_data.csv"
RESULTS_DIR = "results"
PREDICTION_PATH = os.path.join(RESULTS_DIR, "ml_temporal_predictions.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "ml_temporal_metrics.txt")


SENSOR_COLUMNS = [
    "co_level",
    "humidity",
    "light",
    "lpg_level",
    "motion",
    "smoke_level",
    "temperature_c",
]


def add_temporal_features(df):
    work = df.copy()

    work = work.sort_values(["device", "elapsed_seconds"]).copy()

    delta_cols = []
    for col in SENSOR_COLUMNS:
        new_col = f"{col}_abs_delta"
        work[new_col] = work.groupby("device")[col].diff().abs()
        delta_cols.append(new_col)

    work["sensor_change_sum"] = work[delta_cols].sum(axis=1)

    work["same_light_as_previous"] = (
        work.groupby("device")["light"].diff().fillna(1).abs() == 0
    ).astype(int)

    work["same_motion_as_previous"] = (
        work.groupby("device")["motion"].diff().fillna(1).abs() == 0
    ).astype(int)

    signature_cols = [
        "co_level",
        "humidity",
        "light",
        "lpg_level",
        "motion",
        "smoke_level",
        "temperature_c",
    ]

    rounded_signature = (
        work[signature_cols]
        .round(5)
        .astype(str)
        .agg("|".join, axis=1)
    )

    work["same_signature_as_previous"] = (
        rounded_signature.groupby(work["device"]).shift(1) == rounded_signature
    ).astype(int)

    work["duplicate_signature_seen_before"] = (
        rounded_signature.groupby(work["device"]).transform(
            lambda x: x.duplicated(keep="first")
        )
    ).astype(int)

    work["low_change_flag"] = (work["sensor_change_sum"] < 1e-6).astype(int)

    work = work.sort_index().copy()
    work[delta_cols] = work[delta_cols].fillna(0)

    temporal_cols = [
        *delta_cols,
        "sensor_change_sum",
        "same_light_as_previous",
        "same_motion_as_previous",
        "same_signature_as_previous",
        "duplicate_signature_seen_before",
        "low_change_flag",
    ]

    return work, temporal_cols


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    missing = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    needed = ["device", "elapsed_seconds", "attack_label"]
    missing_needed = [col for col in needed if col not in df.columns]
    if missing_needed:
        raise ValueError(f"Missing required columns: {missing_needed}")

    feature_df, temporal_cols = add_temporal_features(df)

    feature_columns = SENSOR_COLUMNS + temporal_cols

    X = feature_df[feature_columns].copy()
    y = feature_df["attack_label"].astype(int)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        feature_df.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

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

    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    lines = []
    lines.append("Temporal machine-learning IoT anomaly detector")
    lines.append("")
    lines.append(f"Input file: {INPUT_PATH}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Training samples: {len(X_train)}")
    lines.append(f"Test samples: {len(X_test)}")
    lines.append("")
    lines.append("Model:")
    lines.append("RandomForestClassifier with snapshot and temporal features")
    lines.append("")
    lines.append("Base sensor features:")
    for col in SENSOR_COLUMNS:
        lines.append(f"- {col}")
    lines.append("")
    lines.append("Temporal features:")
    for col in temporal_cols:
        lines.append(f"- {col}")
    lines.append("")
    lines.append("Confusion matrix [normal, attack]:")
    lines.append(str(cm))
    lines.append("")
    lines.append(f"Accuracy:  {acc:.4f}")
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall:    {recall:.4f}")
    lines.append(f"F1-score:  {f1:.4f}")
    lines.append("")
    lines.append("Top feature importance:")
    for name, value in feature_importance.head(12).items():
        lines.append(f"- {name}: {value:.4f}")

    if "attack_type" in feature_df.columns:
        result = feature_df.loc[test_idx, ["attack_type"]].copy()
        result["y_true"] = y_test.to_numpy()
        result["y_pred"] = y_pred

        attack_rows = result[result["y_true"] == 1]
        rates = attack_rows.groupby("attack_type")["y_pred"].mean().sort_values(ascending=False)

        lines.append("")
        lines.append("Detection rate by attack type on test set:")
        for attack_type, rate in rates.items():
            count = int((attack_rows["attack_type"] == attack_type).sum())
            lines.append(f"- {attack_type}: {rate:.4f} ({count} samples)")

    output = "\n".join(lines)
    print(output)

    pred_df = feature_df.loc[test_idx].copy()
    pred_df["ml_temporal_prediction"] = y_pred
    pred_df.to_csv(PREDICTION_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print("")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
