# -*- coding: utf-8 -*-
"""Generate evaluation plots (confusion matrix, ROC, feature importance)
for a trained beacon/C2 model.

Example
-------
python scripts/plot_metrics.py \
    --model-dir artifacts/xgb_sample \
    --test-data artifacts/iot23_features.parquet
"""
import argparse
import pathlib
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay
)

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
    df_encoded = df_encoded[train_cols]  # ensure correct column order
    return df_encoded

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics for a trained model")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--test-data", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("reports"))
    parser.add_argument("--sample", type=int, default=100_000, help="If set, samples N rows from test data to reduce memory usage.")
    args = parser.parse_args()

    model = joblib.load(args.model_dir / "model.joblib")
    train_cols = joblib.load(args.model_dir / "feature_columns.joblib")
    df = pd.read_parquet(args.test_data)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42)

    y_true = (df["label"].str.lower() == "malicious").astype(int)
    drop_cols = ["ts", "uid", "label", "detailed_label", "id.orig_h", "id.resp_h"]
    X = build_feature_matrix(df, drop_cols, train_cols)

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malicious"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(args.output_dir / "roc_curve.png", dpi=200)
    plt.close()

    # Feature importance
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

    print(f"✅ Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
