# -*- coding: utf-8 -*-
"""Evaluate a trained beacon/C2 detection model on new Zeek‑feature Parquet data.

Example
-------
python scripts/evaluate.py \
    --model-dir artifacts/xgb_sample \
    --test-data artifacts/custom_features.parquet \
    --output-csv reports/custom_scores.csv
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

###############################################################################
# 1 ‑‑ Feature builders (same logic as in train.py)
###############################################################################

def build_feature_matrix(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """One‑hot encode low‑cardinality categoricals; pass numeric unchanged."""
    df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = [
        c for c in df_feat.columns
        if (df_feat[c].dtype.name in ("category", "object")) and (df_feat[c].nunique() <= 100)
    ]

    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, dummy_na=True)
    return df_encoded

###############################################################################
# 2 ‑‑ Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate beacon/C2 detector on new data")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path, help="Directory containing model.joblib + feature_columns.joblib")
    parser.add_argument("--test-data", required=True, type=pathlib.Path, help="Parquet file with features produced by preprocess.py")
    parser.add_argument("--output-csv", type=pathlib.Path, help="If given, write dataframe with predictions to this CSV")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  Load model + feature column ordering
    # ------------------------------------------------------------------
    model_path = args.model_dir / "model.joblib"
    cols_path = args.model_dir / "feature_columns.joblib"

    if not model_path.exists() or not cols_path.exists():
        raise SystemExit(f"Model artifacts not found in {args.model_dir}")

    model = joblib.load(model_path)
    train_cols: List[str] = joblib.load(cols_path)

    # ------------------------------------------------------------------
    #  Load test data
    # ------------------------------------------------------------------
    df = pd.read_parquet(args.test_data)
    print(f"Loaded {len(df):,} rows from {args.test_data}")

    # Build feature matrix
    drop_cols = [
        "ts", "uid", "label", "detailed_label", "id.orig_h", "id.resp_h"
    ]
    X = build_feature_matrix(df, drop_cols)

    # Align columns with training schema
    missing_cols = [c for c in train_cols if c not in X.columns]
    extra_cols = [c for c in X.columns if c not in train_cols]

    if missing_cols:
        for c in missing_cols:
            X[c] = 0
    if extra_cols:
        X = X.drop(columns=extra_cols)

    X = X[train_cols]  # same order

    # ------------------------------------------------------------------
    #  Predict
    # ------------------------------------------------------------------
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    df_out = df.copy()
    df_out["pred"] = pred
    df_out["proba"] = proba

    # ------------------------------------------------------------------
    #  Metrics (if ground‑truth present)
    # ------------------------------------------------------------------
    if "label" in df.columns:
        y_true = (df["label"].str.lower() == "malicious").astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
            "n_samples": int(len(y_true)),
            "pos_rate": float(y_true.mean()),
        }
        print("\nMetrics on provided test data:")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")

        with (args.model_dir / "eval_metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"\n✅ Eval metrics saved to {args.model_dir / 'eval_metrics.json'}")
    else:
        print("No 'label' column in test data — skipping metric computation.")

    # ------------------------------------------------------------------
    #  Optional CSV output
    # ------------------------------------------------------------------
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()