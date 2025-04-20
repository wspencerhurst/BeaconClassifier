# -*- coding: utf-8 -*-
"""Train an XGBoost classifier to detect beacon / C2 traffic.

Assumes you have already run `preprocess.py` to create a Parquet file that
contains engineered features and the original Zeek columns.

Example
-------
python scripts/train.py \
    --train artifacts/iot23_features.parquet \
    --model-dir artifacts/xgb_baseline
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

###############################################################################
# 1 ‑‑ Utility functions
###############################################################################

def build_feature_matrix(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Return X (features) ready for XGBoost.

    * Categorical columns are one‑hot encoded with pandas.get_dummies().
    * Numeric columns are passed through unchanged.
    """
    df_feat = df.drop(columns=drop_cols)

    # Identify valid categoricals (low cardinality only)
    cat_cols = [
        c for c in df_feat.columns
        if (df_feat[c].dtype.name in ("category", "object"))
        and (df_feat[c].nunique() <= 100)
    ]

    print("One-hot encoding:", cat_cols)

    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, dummy_na=True)
    return df_encoded


def compute_class_weight(y: pd.Series) -> float:
    """Return scale_pos_weight for XGBoost (neg / pos)."""
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)

###############################################################################
# 2 ‑‑ Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Train beacon/C2 detector (XGBoost)")
    parser.add_argument("--train", required=True, type=pathlib.Path, help="Parquet features file")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path, help="Directory to save model & metadata")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for hold‑out test set (default 0.2)")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    #  Load data
    # ---------------------------------------------------------------------
    df = pd.read_parquet(args.train)
    print(f"Loaded {len(df):,} rows from {args.train}")

    # Optional downsampling for testing
    df = df.sample(n=500_000, random_state=args.random_seed)

    # Binary label: 1 for Malicious (any C&C / HeartBeat / custom), else 0
    y = (df["label"].str.lower() == "malicious").astype(int)

    # Drop columns that shouldn’t feed the model
    drop_cols = [
        "ts",
        "uid",
        "label",
        "detailed_label",  # keep for analysis but not features
        "id.orig_h",
        "id.resp_h",
    ]

    X = build_feature_matrix(df, drop_cols)
    print(f"Feature matrix shape: {X.shape}")

    # ---------------------------------------------------------------------
    #  Split train/test
    # ---------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    # ---------------------------------------------------------------------
    #  Model & hyper‑params (mirrors winning notebook settings)
    # ---------------------------------------------------------------------
    scale_pos_weight = compute_class_weight(y_train)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_seed,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    #  Evaluation
    # ---------------------------------------------------------------------
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
    }

    print("\nTest‑set metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")

    # ---------------------------------------------------------------------
    #  Persist artifacts
    # ---------------------------------------------------------------------
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "model.joblib")
    joblib.dump(X.columns.tolist(), args.model_dir / "feature_columns.joblib")

    with (args.model_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"\n✅ Model & metrics saved under {args.model_dir}\n")


if __name__ == "__main__":
    main()
