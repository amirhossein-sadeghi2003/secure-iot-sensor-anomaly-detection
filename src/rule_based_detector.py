import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


INPUT_PATH = "data/processed/labeled_iot_spoofing_data.csv"
RESULTS_DIR = "results"
PREDICTION_PATH = os.path.join(RESULTS_DIR, "rule_based_predictions.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "rule_based_metrics.txt")


def find_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def make_binary_label(df):
    label_col = find_column(
        df.columns,
        ["is_attack", "attack", "attack_label", "label", "anomaly", "target"],
    )

    if label_col is not None:
        values = df[label_col]
        if pd.api.types.is_numeric_dtype(values):
            return values.astype(int), label_col

        text = values.astype(str).str.lower().str.strip()
        y = text.isin(["1", "true", "attack", "anomaly", "spoofed", "spoofing"])
        return y.astype(int), label_col

    attack_type_col = find_column(
        df.columns,
        ["attack_type", "anomaly_type", "spoofing_type", "scenario"],
    )

    if attack_type_col is None:
        raise ValueError("No attack label column was found in the dataset.")

    text = df[attack_type_col].astype(str).str.lower().str.strip()
    y = ~text.isin(["normal", "none", "0", "clean"])
    return y.astype(int), attack_type_col


def range_rule(values, baseline, low_q=0.002, high_q=0.998):
    low = baseline.quantile(low_q)
    high = baseline.quantile(high_q)
    iqr = baseline.quantile(0.75) - baseline.quantile(0.25)

    if pd.isna(iqr) or iqr == 0:
        margin = 0
    else:
        margin = 0.35 * iqr

    return (values < low - margin) | (values > high + margin)


def device_range_rule(df, normal_df, col, device_col):
    rule = pd.Series(False, index=df.index)

    for device, normal_part in normal_df.groupby(device_col):
        if len(normal_part) < 30:
            continue

        idx = df[df[device_col] == device].index
        rule.loc[idx] = range_rule(df.loc[idx, col], normal_part[col])

    return rule


def replay_duplicate_rule(df, sensor_cols, device_col, time_col):
    rule = pd.Series(False, index=df.index)

    ordered = df.sort_values([device_col, time_col]).copy()
    signature = ordered[sensor_cols].round(6).astype(str).agg("|".join, axis=1)

    temp = pd.DataFrame({
        "device": ordered[device_col],
        "signature": signature,
    }, index=ordered.index)

    duplicate_later = temp.groupby("device")["signature"].transform(
        lambda x: x.duplicated(keep="first")
    )

    rule.loc[ordered.index] = duplicate_later.to_numpy()
    return rule


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    y_true, label_source = make_binary_label(df)

    normal_df = df[y_true == 0].copy()
    if normal_df.empty:
        raise ValueError("No normal samples were found for building baseline rules.")

    columns = set(df.columns)

    temp_col = find_column(columns, ["temperature_c", "temperature", "temp"])
    humidity_col = find_column(columns, ["humidity"])
    co_col = find_column(columns, ["co_level", "co", "carbon_monoxide"])
    lpg_col = find_column(columns, ["lpg_level", "lpg"])
    smoke_col = find_column(columns, ["smoke_level", "smoke"])
    light_col = find_column(columns, ["light"])
    motion_col = find_column(columns, ["motion"])
    device_col = find_column(columns, ["device", "device_id"])
    time_col = find_column(columns, ["elapsed_seconds", "device_time_step", "timestamp", "ts"])
    attack_type_col = find_column(columns, ["attack_type", "anomaly_type", "spoofing_type", "scenario"])

    numeric_sensor_cols = [
        col for col in [temp_col, humidity_col, co_col, lpg_col, smoke_col]
        if col is not None
    ]

    rules = pd.DataFrame(index=df.index)

    for col in numeric_sensor_cols:
        if device_col is not None:
            rules[f"{col}_outside_device_range"] = device_range_rule(df, normal_df, col, device_col)
        else:
            rules[f"{col}_outside_global_range"] = range_rule(df[col], normal_df[col])

    gas_cols = [col for col in [co_col, lpg_col, smoke_col] if col is not None]
    if len(gas_cols) >= 2:
        gas_flags = []
        for col in gas_cols:
            high_limit = normal_df[col].quantile(0.998)
            gas_flags.append(df[col] > high_limit)

        high_gas_count = np.vstack([flag.to_numpy() for flag in gas_flags]).sum(axis=0)
        rules["multiple_gas_sensor_spike"] = high_gas_count >= 2

    if light_col is not None and motion_col is not None:
        rules["light_motion_mismatch"] = (
            (df[light_col].astype(int) == 0) &
            (df[motion_col].astype(int) == 1)
        )

    range_rule_cols = [
        col for col in rules.columns
        if col.endswith("_outside_device_range") or col.endswith("_outside_global_range")
    ]

    range_count = rules[range_rule_cols].sum(axis=1)

    final_rule = range_count >= 2

    if "multiple_gas_sensor_spike" in rules.columns:
        final_rule = final_rule | rules["multiple_gas_sensor_spike"]

    if "light_motion_mismatch" in rules.columns:
        final_rule = final_rule | rules["light_motion_mismatch"]

    rules["final_decision"] = final_rule

    y_pred = final_rule.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    lines = []
    lines.append("Rule-based IoT anomaly detector")
    lines.append("")
    lines.append(f"Input file: {INPUT_PATH}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Label source: {label_source}")
    lines.append(f"Normal samples: {(y_true == 0).sum()}")
    lines.append(f"Attack samples: {(y_true == 1).sum()}")
    lines.append(f"Predicted normal: {(y_pred == 0).sum()}")
    lines.append(f"Predicted attack: {(y_pred == 1).sum()}")
    lines.append("")
    lines.append("Confusion matrix [normal, attack]:")
    lines.append(str(cm))
    lines.append("")
    lines.append(f"Accuracy:  {acc:.4f}")
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall:    {recall:.4f}")
    lines.append(f"F1-score:  {f1:.4f}")
    lines.append("")
    lines.append("Triggered rules:")
    for col in rules.columns:
        lines.append(f"- {col}: {int(rules[col].sum())}")

    if attack_type_col is not None:
        lines.append("")
        lines.append("Detection rate by attack type:")
        temp = pd.DataFrame({
            "attack_type": df[attack_type_col],
            "y_true": y_true,
            "y_pred": y_pred,
        })
        attack_rows = temp[temp["y_true"] == 1]
        rates = attack_rows.groupby("attack_type")["y_pred"].mean().sort_values(ascending=False)

        for attack_type, rate in rates.items():
            count = (attack_rows["attack_type"] == attack_type).sum()
            lines.append(f"- {attack_type}: {rate:.4f} ({count} samples)")

    output = "\n".join(lines)

    print(output)

    result_df = df.copy()
    result_df["rule_based_prediction"] = y_pred
    result_df.to_csv(PREDICTION_PATH, index=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print("")
    print(f"Saved predictions to: {PREDICTION_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
