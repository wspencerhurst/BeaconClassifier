# -*- coding: utf-8 -*-
import argparse
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Prepare combined training and hold-out test datasets.")
    parser.add_argument("--iot23-input", required=True, type=pathlib.Path,
                        help="Path to preprocessed IOT-23 Parquet file.")
    parser.add_argument("--custom-input", required=True, type=pathlib.Path,
                        help="Path to preprocessed custom C2 Parquet file.")
    parser.add_argument("--combined-train-output", required=True, type=pathlib.Path,
                        help="Output path for the combined training Parquet file.")
    parser.add_argument("--custom-holdout-output", required=True, type=pathlib.Path,
                        help="Output path for the custom C2 hold-out test Parquet file.")
    parser.add_argument("--custom-train-fraction", type=float, default=0.7,
                        help="Fraction of custom C2 data to use for training (rest is for hold-out). Default 0.7.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducible train/test split of custom data.")
    
    args = parser.parse_args()

    args.combined_train_output.parent.mkdir(parents=True, exist_ok=True)
    args.custom_holdout_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading IOT-23 data from: {args.iot23_input}")
    try:
        df_iot23 = pd.read_parquet(args.iot23_input)
        print(f"Loaded {len(df_iot23):,} rows from IOT-23 data.")
    except Exception as e:
        raise SystemExit(f"Error loading IOT-23 data: {e}")

    print(f"Loading custom C2 data from: {args.custom_input}")
    try:
        df_custom_all = pd.read_parquet(args.custom_input)
        print(f"Loaded {len(df_custom_all):,} rows from custom C2 data.")
    except Exception as e:
        raise SystemExit(f"Error loading custom C2 data: {e}")

    if df_custom_all.empty:
        raise SystemExit("Custom C2 DataFrame is empty. Cannot proceed.")

    if 'is_c2' not in df_custom_all.columns:
        print("Warning: 'is_c2' column not found in custom data. Attempting to derive from 'label'.")
        if 'label' in df_custom_all.columns:
            df_custom_all['is_c2'] = (df_custom_all['label'].astype(str).str.contains('Malicious', case=False, na=False)).astype(int)
            print("Derived 'is_c2' from 'label' for custom data stratification.")
        else:
            raise SystemExit("Error: 'is_c2' column missing in custom data and 'label' column also missing. Cannot stratify split.")
    
    if df_custom_all['is_c2'].nunique() < 2:
        print("Warning: Custom data 'is_c2' column has less than 2 unique values. Stratification might not be effective or possible.")
        try:
            df_custom_train, df_custom_holdout_test = train_test_split(
                df_custom_all,
                train_size=args.custom_train_fraction,
                random_state=args.random_seed,
                shuffle=True
            )
        except ValueError as e:
             if len(df_custom_all) * args.custom_train_fraction < 1 or len(df_custom_all) * (1-args.custom_train_fraction) < 1:
                 print(f"Custom dataset too small for specified split fraction. Using all for training, creating empty holdout or vice-versa. Error: {e}")
                 if args.custom_train_fraction > 0.5:
                     df_custom_train = df_custom_all.copy()
                     df_custom_holdout_test = pd.DataFrame(columns=df_custom_all.columns)
                 else:
                     df_custom_train = pd.DataFrame(columns=df_custom_all.columns)
                     df_custom_holdout_test = df_custom_all.copy()
             else:
                 raise e

    else:
        df_custom_train, df_custom_holdout_test = train_test_split(
            df_custom_all,
            train_size=args.custom_train_fraction,
            random_state=args.random_seed,
            stratify=df_custom_all['is_c2'],
            shuffle=True
        )

    print(f"Split custom data: {len(df_custom_train):,} for training, {len(df_custom_holdout_test):,} for hold-out testing.")

    print("Combining IOT-23 data with the training portion of custom C2 data...")
    df_combined_train = pd.concat([df_iot23, df_custom_train], ignore_index=True, sort=False)
    print(f"Combined training dataset has {len(df_combined_train):,} rows.")

    print(f"Saving combined training data to: {args.combined_train_output}")
    df_combined_train.to_parquet(args.combined_train_output, index=False, compression="snappy")

    if not df_custom_holdout_test.empty:
        print(f"Saving custom C2 hold-out test data to: {args.custom_holdout_output}")
        df_custom_holdout_test.to_parquet(args.custom_holdout_output, index=False, compression="snappy")
    else:
        print(f"Custom C2 hold-out test set is empty. Not saving. Check custom_train_fraction and dataset size.")

    print("\nSuccessfully prepared combined training and custom hold-out test datasets.")
    print(f"  Combined training data: {args.combined_train_output}")
    if not df_custom_holdout_test.empty:
        print(f"  Custom hold-out test data: {args.custom_holdout_output}")

if __name__ == "__main__":
    main()