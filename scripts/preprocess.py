# -*- coding: utf-8 -*-
"""Pre‑process Zeek conn.log files for Beacon/C2 detection.

Usage (example):
    python scripts/preprocess.py \
        --input-dir data/iot23_raw \
        --output-fp artifacts/iot23_features.parquet

Steps
-----
1.  Iterates over every file found in --input-dir (recursive OK).
2.  Reads pipe‑delimited Zeek logs **with or without** the leading header line.
3.  Cleans & converts dtypes (timestamps, ports, numerics, categorical dtypes).
4.  Adds baseline connection features **+** simple beacon‑oriented temporal
    features (inter‑arrival time stats & jitter for each (src_ip, dst_ip, dst_port)).
5.  Writes the resulting DataFrame to Apache Parquet so later stages load fast.

The script is intentionally dependency‑light: pandas, numpy, pyarrow, tqdm.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

###############################################################################
# 1 ‑‑ Configuration
###############################################################################

# Explicit column list because some Zeek files in IoT‑23 have the header stripped
COLUMNS: List[str] = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "label",
    "detailed_label",  # IoT‑23 sometimes uses dashed name; we’ll normalise below
]

NUMERIC_COLS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

IAT_KEY = ["id.orig_h", "id.resp_h", "id.resp_p"]  # granularity for IAT features

###############################################################################
# 2 ‑‑ Helpers
###############################################################################

def _read_zeek(path: pathlib.Path) -> pd.DataFrame:
    """Read a single Zeek conn.log (pipe‑delimited) into a DataFrame."""
    # Sniff whether file contains a header starting with #fields
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        first_line = fh.readline()
    header = None
    if first_line.startswith("#"):
        # There *is* a header. Use pandas’ comment handling to skip # lines.
        header = 0  # pandas will use the #fields row as header → we'll rename later
    
    df = pd.read_csv(
        path,
        sep="|",
        header=header,
        comment="#",
        names=COLUMNS if header is None else None,
        na_values="-",
        low_memory=False,
    )

    # Normalise detailed‑label column name if needed
    if "detailed-label" in df.columns and "detailed_label" not in df.columns:
        df = df.rename(columns={"detailed-label": "detailed_label"})

    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate dtypes in‑place."""
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ports are small ints; helps memory & speeds groupby
    for col in ("id.orig_p", "id.resp_p"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    # Strings → category (optional memory win)
    for cat in ("proto", "service", "conn_state", "label", "detailed_label"):
        if cat in df.columns:
            df[cat] = df[cat].astype("category")

    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add inter‑arrival time (IAT) statistics & jitter per flow‑key."""
    df = df.sort_values("ts")

    # Seconds between starts of consecutive connections for same key
    df["iat"] = (
        df.groupby(IAT_KEY)["ts"].diff().dt.total_seconds().fillna(0)
    )

    grp = df.groupby(IAT_KEY)["iat"]
    df["iat_mean"] = grp.transform("mean")
    df["iat_std"] = grp.transform("std").fillna(0)
    df["iat_jitter"] = df["iat_std"] / df["iat_mean"].replace(0, np.nan)

    return df

###############################################################################
# 3 ‑‑ Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Pre‑process Zeek conn.log files.")
    parser.add_argument("--input-dir", required=True, type=pathlib.Path,
                        help="Directory containing .log/.csv files (recursively searched).")
    parser.add_argument("--output-fp", required=True, type=pathlib.Path,
                        help="Output Parquet file path.")
    parser.add_argument("--recursive", action="store_true", default=False,
                        help="If set, walk sub‑directories under --input-dir.")

    args = parser.parse_args()

    files = (
        list(args.input_dir.rglob("*.log")) if args.recursive else list(args.input_dir.glob("*.log"))
    )
    files += list(args.input_dir.rglob("*.csv")) if args.recursive else list(args.input_dir.glob("*.csv"))

    if not files:
        raise SystemExit(f"No .log/.csv files found in {args.input_dir}")


    # Prepare for streaming write
    args.output_fp.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0

    for fp in tqdm(files, desc="Processing Zeek logs"):
        df = _read_zeek(fp)
        df = _coerce_types(df)
        df = _add_temporal_features(df)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output_fp, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(df)

    if writer is not None:
        writer.close()

    print(f"✅ Saved {total_rows:,} rows → {args.output_fp}")




if __name__ == "__main__":
    main()
