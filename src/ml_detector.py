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
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/labeled_iot_spoofing_data.csv"
RESULTS_DIR = "results"
PREDICTION_PATH = os.path.join(RESULTS_DIR, "ml_detector_predictions.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "ml_detector_metrics.txt")


FEATURE_COLUMNS = [
    "co_level",
    "humidity",
    "light",
    "lpg_level",
    "motion",
    "smoke_level",
    "temperature_c",
]


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if "attack_label" not in df.columns:
        raise ValueError("Expected column was not found: attack_label")

    X = df[FEATURE_COLUMNS].copy()
    y = df["attack_label"].astype(int)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
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
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)

    lines = []
    lines.append("Machine-learning IoT anomaly detector")
    lines.append("")
    lines.append(f"Input file: {INPUT_PATH}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Training samples: {len(X_train)}")
    lines.append(f"Test samples: {len(X_test)}")
    lines.append("")
    lines.append("Model:")
    lines.append("RandomForestClassifier")
    lines.append("")
    lines.append("Feature columns:")
    for col in FEATURE_COLUMNS:
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
    lines.append("Feature importance:")
    for name, value in feature_importance.items():
        lines.append(f"- {name}: {value:.4f}")

    if "attack_type" in df.columns:
        result = df.loc[test_idx, ["attack_type"]].copy()
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

    pred_df = df.loc[test_idx].copy()
    pred_df["ml_prediction"] = y_pred
    pred_df.to_csv(PREDICTION_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print("")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
