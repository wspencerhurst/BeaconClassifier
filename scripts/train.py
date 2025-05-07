# -*- coding: utf-8 -*-
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

def build_feature_matrix(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    df_feat = df.drop(columns=drop_cols)

    cat_cols = [
        c for c in df_feat.columns
        if (df_feat[c].dtype.name in ("category", "object"))
        and (df_feat[c].nunique() <= 100)
    ]

    print("One-hot encoding:", cat_cols)

    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, dummy_na=True)
    return df_encoded


def compute_class_weight(y: pd.Series) -> float:
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / max(n_pos, 1)


def print_metrics(title, y_true, y_pred, y_proba):
    print(f"\n=== {title} ===")
    print(f" Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f" Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f" Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f" F1-score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f" ROC‑AUC  : {roc_auc_score(y_true, y_proba):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train beacon/C2 detector (XGBoost)")
    parser.add_argument("--train", required=True, type=pathlib.Path, help="Parquet features file")
    parser.add_argument("--model-dir", required=True, type=pathlib.Path, help="Directory to save model & metadata")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for hold‑out test set (default 0.2)")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.train)
    print(f"Loaded {len(df):,} rows from {args.train}")

    df = df.sample(n=500_000, random_state=args.random_seed)

    y = (df["label"].str.lower() == "malicious").astype(int)

    drop_cols = [
        "ts",
        "uid",
        "label",
        "detailed_label",
        "id.orig_h",
        "id.resp_h",
        "is_c2",
    ]

    X = build_feature_matrix(df, drop_cols)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    scale_mal = compute_class_weight(y_train)
    model_mal = XGBClassifier(
        n_estimators=400, learning_rate=0.1, max_depth=6,
        subsample=0.9, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        n_jobs=-1, scale_pos_weight=scale_mal,
        random_state=args.random_seed,
        tree_method="hist",
    )
    model_mal.fit(X_train, y_train)

    y_pred_prob_mal = model_mal.predict_proba(X_test)[:, 1]
    y_pred_mal      = (y_pred_prob_mal >= 0.5).astype(int)
    print_metrics("Malicious vs Benign", y_test, y_pred_mal, y_pred_prob_mal)

    y_c2 = df["is_c2"]
    X_c2 = X

    Xc2_train, Xc2_test, yc2_train, yc2_test = train_test_split(
        X_c2, y_c2, test_size=args.test_size,
        random_state=args.random_seed, stratify=y_c2
    )

    scale_c2 = compute_class_weight(yc2_train)

    model_c2 = XGBClassifier(
        n_estimators=400, learning_rate=0.1, max_depth=6,
        subsample=0.9, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        n_jobs=-1, scale_pos_weight=scale_c2,
        random_state=args.random_seed,
        tree_method="hist",
    )
    model_c2.fit(Xc2_train, yc2_train)

    yc2_prob = model_c2.predict_proba(Xc2_test)[:, 1]
    yc2_pred = (yc2_prob >= 0.5).astype(int)
    print_metrics("C2 vs Everything", yc2_test, yc2_pred, yc2_prob)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_mal, args.model_dir / "model_mal.joblib")
    joblib.dump(model_c2,  args.model_dir / "model_c2.joblib")
    joblib.dump(X.columns.tolist(), args.model_dir / "feature_columns.joblib")

    print(f"\nSaved dual models to {args.model_dir}\n")


if __name__ == "__main__":
    main()
