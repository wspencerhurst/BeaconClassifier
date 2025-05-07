# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import gzip 

COLUMNS_IOT23: List[str] = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed_label",
]

NUMERIC_COLS = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]

IAT_KEY = ["id.orig_h", "id.resp_h", "id.resp_p"]

def _read_iot23_zeek(path: pathlib.Path) -> pd.DataFrame:
    pandas_header_arg: Optional[int] = None
    pandas_names_arg: Optional[List[str]] = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh_peek:
            first_line_content = fh_peek.readline().strip()
    except Exception as e:
        print(f"Warning: Could not read first line of {path}: {e}")
        return pd.DataFrame()

    if not first_line_content:
        print(f"Info: Skipping empty file {path}")
        return pd.DataFrame()

    if first_line_content.startswith("#"):
        pandas_header_arg = 0
    else:
        pandas_names_arg = COLUMNS_IOT23
    
    try:
        df = pd.read_csv(
            path,
            sep="|",
            header=pandas_header_arg,
            names=pandas_names_arg,
            comment="#",
            na_values=["-", "(empty)"],
            low_memory=False,
            dtype=str
        )
    except Exception as e:
        print(f"Error reading CSV {path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    if "detailed-label" in df.columns and "detailed_label" not in df.columns:
        df = df.rename(columns={"detailed-label": "detailed_label"})

    for col in COLUMNS_IOT23:
        if col not in df.columns:
            df[col] = pd.NA

    if pandas_names_arg is not None:
        if len(df.columns) > len(pandas_names_arg):
            df = df.iloc[:, :len(pandas_names_arg)]
        df.columns = pandas_names_arg[:len(df.columns)]

    return df

def _read_custom_zeek(path: pathlib.Path) -> pd.DataFrame:
    _column_names = []
    _separator = '\t'
    _num_header_lines_to_skip = 0

    fopen = gzip.open if path.suffix == ".gz" else open
    
    with fopen(path, "rt", encoding="utf-8", errors="ignore") as fh_scan:
        for idx, line_content in enumerate(fh_scan):
            stripped_line = line_content.strip()
            if not stripped_line.startswith("#"):
                _num_header_lines_to_skip = idx
                break 
            if stripped_line.startswith("#fields"):
                _column_names = stripped_line.split('\t')[1:]
            elif stripped_line.startswith("#separator"):
                _separator = "\t" 
            _num_header_lines_to_skip = idx + 1
    
    if not _column_names:
        print(f"Warning: Could not find #fields line in {path}. Attempting to use default schema.")
        _column_names = [col for col in COLUMNS_IOT23 if col not in ["label", "detailed_label"]]
        if not _column_names:
             raise ValueError(f"Critcal: Could not determine column names for {path}")


    df = pd.read_csv(
        path,
        sep=_separator,
        names=_column_names,
        skiprows=_num_header_lines_to_skip,
        na_values=["-", "(empty)"],
        low_memory=False,
        compression='gzip' if path.suffix == ".gz" else None,
    )
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("id.orig_p", "id.resp_p"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
        else:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Int32")

    for bool_col in ("local_orig", "local_resp"):
        if bool_col in df.columns:
            s_bool = df[bool_col].copy()
            if s_bool.dtype == 'object' or pd.api.types.is_string_dtype(s_bool):
                s_bool_str_upper = s_bool.str.upper().fillna("THIS_IS_NA")
                
                mapping = {
                    "T": True, "TRUE": True, "1": True, "Y": True,
                    "F": False, "FALSE": False, "0": False, "N": False,
                    "THIS_IS_NA": pd.NA,
                    "": pd.NA, "NONE": pd.NA, "NULL": pd.NA, 
                }
                df[bool_col] = s_bool_str_upper.map(mapping)
            elif pd.api.types.is_bool_dtype(s_bool) or pd.api.types.is_numeric_dtype(s_bool):
                 mapping_numeric_bool = {1: True, 0: False, 1.0: True, 0.0: False}
                 if s_bool.dtype != 'boolean':
                     df[bool_col] = s_bool.map(mapping_numeric_bool)


            try:
                df[bool_col] = df[bool_col].astype("boolean")
            except TypeError as e:
                print(f"Warning: Could not convert column {bool_col} to boolean. Contents: {df[bool_col].unique()[:5]}. Error: {e}")
                df[bool_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        else:
            df[bool_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    for cat_col in ("proto", "service", "conn_state", "label", "detailed_label", "history"):
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("MISSING_CAT").astype("category")
        else:
            df[cat_col] = pd.Series("MISSING_CAT", index=df.index, dtype="category")
            
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not all(col in df.columns for col in ["ts"] + IAT_KEY):
        print(f"Skipping temporal features: one or more key columns missing (ts, {IAT_KEY})")
        for col in ["iat", "iat_mean", "iat_std", "iat_jitter"]:
            df[col] = 0.0
        return df
        
    df = df.sort_values("ts")
    df["iat"] = (
        df.groupby(IAT_KEY)["ts"].diff().dt.total_seconds().fillna(0)
    )
    grp = df.groupby(IAT_KEY)["iat"]
    df["iat_mean"] = grp.transform("mean")
    df["iat_std"] = grp.transform("std").fillna(0)
    df["iat_jitter"] = df["iat_std"] / df["iat_mean"].replace(0, np.nan) 
    df["iat_jitter"] = df["iat_jitter"].fillna(0)

    return df

def main():
    parser = argparse.ArgumentParser(description="Pre‑process Zeek conn.log files.")
    parser.add_argument("--input-dir", required=True, type=pathlib.Path,
                        help="Directory containing log files.")
    parser.add_argument("--output-fp", required=True, type=pathlib.Path,
                        help="Output Parquet file path.")
    parser.add_argument("--input-type", choices=["iot23", "custom"], default="iot23",
                        help="Format of input logs: 'iot23' (pipe-sep) or 'custom' (tab-sep, .gz).")
    parser.add_argument("--victim-ip", type=str, default=None,
                        help="For 'custom' input: IP of the victim machine running the beacon.")
    parser.add_argument("--c2-server-ip", type=str, default=None,
                        help="For 'custom' input: IP of the C2 server.")
    parser.add_argument("--recursive", action="store_true", default=False,
                        help="If set, walk sub‑directories under --input-dir.")
    args = parser.parse_args()

    if args.input_type == "custom" and (not args.victim_ip or not args.c2_server_ip):
        raise SystemExit("For --input-type 'custom', --victim-ip and --c2-server-ip are required.")

    files = []
    if args.input_type == "custom":
        pattern = "conn.*.log.gz"
        base_path = args.input_dir
        files = list(base_path.rglob(pattern)) if args.recursive else list(base_path.glob(pattern))
    else:
        patterns = ["*.log", "*.csv"] 
        base_path = args.input_dir
        for p_str in patterns:
            if args.recursive:
                files.extend(base_path.rglob(p_str))
            else:
                files.extend(base_path.glob(p_str))
    
    files = sorted(list(set(files))) 

    if not files:
        if args.input_type == "custom":
             used_pattern = f"'{pattern}' in '{args.input_dir}'"
        else:
             used_pattern = f"{patterns} in '{args.input_dir}'"
        raise SystemExit(f"No files matching {used_pattern} (recursive={args.recursive}) for type '{args.input_type}'")


    args.output_fp.parent.mkdir(parents=True, exist_ok=True)
    all_dfs = [] 
    total_rows_processed = 0

    for fp in tqdm(files, desc=f"Processing {args.input_type} Zeek logs"):
        current_df = None
        try:
            if args.input_type == "custom":
                current_df = _read_custom_zeek(fp)
                if current_df.empty:
                    print(f"Info: Skipping empty or unreadable custom file: {fp}")
                    continue
                
                current_df["label"] = "Benign"
                current_df["detailed_label"] = "-"
                
                if 'id.orig_h' in current_df.columns and 'id.resp_h' in current_df.columns:
                    c2_mask = (
                        (current_df['id.orig_h'] == args.victim_ip) & (current_df['id.resp_h'] == args.c2_server_ip)
                    ) | (
                        (current_df['id.orig_h'] == args.c2_server_ip) & (current_df['id.resp_h'] == args.victim_ip)
                    )
                    current_df.loc[c2_mask, 'label'] = 'Malicious'
                    current_df.loc[c2_mask, 'detailed_label'] = 'C&C_Custom'
                    current_df["is_c2"] = 0 
                    current_df.loc[c2_mask, "is_c2"] = 1
                else:
                    print(f"Warning: IP columns missing in custom log {fp}, cannot apply C2 labeling. Defaulting is_c2 to 0.")
                    current_df["is_c2"] = 0
            
            else:
                current_df = _read_iot23_zeek(fp)
                if current_df.empty:
                    print(f"Info: Skipping empty or unreadable IOT23 file: {fp}")
                    continue

                if "label" not in current_df.columns: current_df["label"] = "" 
                if "detailed_label" not in current_df.columns: current_df["detailed_label"] = ""
                
                current_df = current_df.copy() 

                label_str_upper = current_df["label"].astype(str).str.upper().fillna('')
                detailed_str_upper = current_df["detailed_label"].astype(str).str.upper().fillna('')

                keywords = ["C&C", "HEARTBEAT", "FILEDOWNLOAD"] 
                
                current_df["is_c2"] = 0 
                for kw in keywords:
                    mask_label = label_str_upper.str.contains(kw, regex=False, na=False)
                    mask_detailed = detailed_str_upper.str.contains(kw, regex=False, na=False)
                    current_df.loc[mask_label | mask_detailed, "is_c2"] = 1
            
            current_df = _coerce_types(current_df)
            
            current_df = _add_temporal_features(current_df)

            if 'is_c2' in current_df.columns:
                current_df['is_c2'] = current_df['is_c2'].fillna(0).astype(int)
            else:
                current_df['is_c2'] = 0 
            
            all_dfs.append(current_df)
            total_rows_processed += len(current_df)

        except Exception as e:
            print(f"CRITICAL Error processing file {fp}: {e}. Skipping this file.")
            import traceback
            traceback.print_exc()
            continue 

    if not all_dfs:
        raise SystemExit("No dataframes were successfully processed. Check input files, paths, and previous errors.")

    final_df = pd.concat(all_dfs, ignore_index=True, sort=False) # sort=False to maintain column order from first df mostly
    if final_df.empty:
        print("No data to save after processing all files.")
    else:
        try:
            table = pa.Table.from_pandas(final_df, preserve_index=False)
            pq.write_table(table, args.output_fp, compression="snappy")
            print(f"Saved {total_rows_processed:,} rows from {len(all_dfs)} processed files → {args.output_fp}")
        except Exception as e:
            print(f"Error writing Parquet file: {e}")
            print("Schema of final_df:")
            print(final_df.info())
            raise


if __name__ == "__main__":
    main()