# -*- coding: utf-8 -*-
import argparse
import pathlib
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np

def build_feature_matrix(df: pd.DataFrame, drop_cols, train_cols) -> pd.DataFrame:
    df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    cat_cols = [
        c for c in df_feat.columns
        if df_feat[c].dtype.name in ("category", "object") and df_feat[c].nunique() <= 100
    ]
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, dummy_na=True)
    for col in train_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded.reindex(columns=train_cols, fill_value=0)
    return df_encoded

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for a trained model")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--test-data", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("reports"))
    parser.add_argument("--sample", type=int, default=100_000, help="If set, samples N rows from test data to reduce memory usage.")
    parser.add_argument("--task", choices=["mal", "c2"], default="mal", help="'mal' = Malicious vs Benign  |  'c2' = C2 vs Everything")
    args = parser.parse_args()

    model_path = (args.model_dir / "model_c2.joblib") if args.task == "c2" \
             else (args.model_dir / "model_mal.joblib")
    model = joblib.load(model_path)
    train_cols = joblib.load(args.model_dir / "feature_columns.joblib")

    df = pd.read_parquet(args.test_data)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)

    if args.task == "c2":
        y_true = df["is_c2"]
    else:
        y_true = (df["label"].str.lower().str.contains("malicious")).astype(int)

    drop_cols = ["ts", "uid", "label", "detailed_label", "id.orig_h", "id.resp_h", "is_c2"]
    
    
    # ...
    X = build_feature_matrix(df, drop_cols, train_cols)

    y_pred_proba = model.predict_proba(X)[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Metrics for Different Thresholds ---")
    thresholds_to_test = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01] # Define your thresholds
    #thresholds_to_test = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001] # Define your thresholds

    # Use the standard 0.5 threshold for the main confusion matrix plot
    y_pred_default_cm = (y_pred_proba >= 0.5).astype(int)

    for t in thresholds_to_test:
        y_pred_t = (y_pred_proba >= t).astype(int)
        print(f"\nMetrics for Threshold: {t:.2f}")
        print(f"  Accuracy : {accuracy_score(y_true, y_pred_t):.4f}")
        print(f"  Precision: {precision_score(y_true, y_pred_t, zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(y_true, y_pred_t, zero_division=0):.4f}")
        print(f"  F1-score : {f1_score(y_true, y_pred_t, zero_division=0):.4f}")
        print(f"  ROC AUC  : {roc_auc_score(y_true, y_pred_proba):.4f}")
        print(f"n_samples: {len(y_true)}")
        print(f"pos_rate: {float(y_true.mean()):.4f}")

    cm = confusion_matrix(y_true, y_pred_default_cm)
    labels = ["Non‑C2", "C2"] if args.task == "c2" else ["Benign", "Malicious"]
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Test-Set Confusion Matrix (Threshold 0.5)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix_thresh_0.5.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.savefig(args.output_dir / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_display.plot()
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(args.output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()
    print("\n-> Plotted Precision-Recall Curve.")

    f1_scores_pr = (2 * precision * recall) / (precision + recall)
    best_f1_idx = np.argmax(f1_scores_pr[:-1])
    best_threshold_f1 = pr_thresholds[best_f1_idx]
    print(f"Threshold for best F1-score ({f1_scores_pr[best_f1_idx]:.4f}) from PR curve: {best_threshold_f1:.4f}")


    booster = model.get_booster()


    labels = ["Non‑C2", "C2"] if args.task == "c2" else ["Benign", "Malicious"]
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.png", dpi=200)
    plt.close()

    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    sorted_imp = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:20])

    plt.figure(figsize=(8, 6))
    plt.barh(list(sorted_imp.keys())[::-1], list(sorted_imp.values())[::-1])
    plt.title("Top 20 Features by Gain")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(args.output_dir / "feature_importance.png", dpi=200)
    plt.close()

    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
