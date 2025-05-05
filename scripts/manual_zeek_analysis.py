import pandas as pd
import pathlib
import sys
from typing import List
import io
import numpy as np
import plotly.express as px
import socket
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft
import math

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
    "detailed_label",
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

# # 7-1 IOT-23 Dataset Group
# SUSPECT_GROUP = [
#     "147.231.100.5",
#     "89.221.210.188",
#     "81.2.248.189",
#     "46.28.110.244",
#     "81.200.60.11",
#     "37.157.198.150",
#     "212.96.160.147",
#     # "185.130.215.13",
#     "80.79.25.111",
# ]

# Our Dataset
SUSPECT_GROUP = [
    "185.125.190.56"
]

IAT_KEY = ["id.orig_h", "id.resp_h", "id.resp_p"]  # granularity for IAT features

def _read_zeek(path: pathlib.Path) -> pd.DataFrame:
    """Read a single Zeek conn.log (tab-delimited) using #fields as headers, no other changes."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except Exception as e:
        raise IOError(f"Failed to read file: {e}")

    # Find the #fields line and extract headers
    fields_line = next((line for line in lines if line.startswith("#fields")), None)
    if not fields_line:
        raise ValueError("No #fields line found in Zeek log.")
    columns = fields_line.strip().split("\t")[1:]

    # Remove comment lines
    data_lines = [line for line in lines if not line.startswith("#")]

    # Read the data with extracted headers
    df = pd.read_csv(
        io.StringIO("".join(data_lines)),
        sep="\t",
        names=columns,
        na_values="-",
        low_memory=False,
    )

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


def plot_histogram(df: pd.DataFrame) -> pd.DataFrame:
    fig = px.histogram(
        data_frame=df,
        x="ts",
        title="Connections Over Time",
        nbins=100  # adjust for granularity
    )

    fig.show()
    
def print_suspect_group_jitter(df):
    suspect_df = df[df["id.resp_h"].isin(SUSPECT_GROUP)]
    suspect_df = suspect_df[["id.resp_h", "iat_jitter"]].drop_duplicates()
    
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 0,
        "display.colheader_justify", "left"
    ):
        print(suspect_df[["id.resp_h", "iat_jitter"]])
        
def print_non_suspect_group_jitter(df):
    non_suspect_df = df[~df["id.resp_h"].isin(SUSPECT_GROUP)]
    non_suspect_df = (
        non_suspect_df[["id.resp_h", "iat_jitter"]]
        .dropna(subset=["iat_jitter"])
        .drop_duplicates()
        .sort_values("iat_jitter", ascending=False)
    )
        
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 0,
        "display.colheader_justify", "left"
    ):
        print(non_suspect_df[["id.resp_h", "iat_jitter"]])
        
        
def get_whois_info(ip):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("whois.iana.org", 43))
            s.sendall((ip + "\r\n").encode())
            response = s.recv(1024).decode()
        
        # Extract referral whois server (like RIPE or ARIN)
        whois_server = None
        for line in response.splitlines():
            if line.lower().startswith("refer:"):
                whois_server = line.split(":")[1].strip()
                break

        if not whois_server:
            return ip, "N/A", "N/A", "N/A"

        # Query actual RIR whois server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((whois_server, 43))
            s.sendall((ip + "\r\n").encode())
            response = b""
            while True:
                chunk = s.recv(2048)
                if not chunk:
                    break
                response += chunk

        org = netname = orgname = "N/A"
        for line in response.decode(errors="ignore").splitlines():
            if "Organization:" in line:
                org = line.split(":", 1)[-1].strip()
            elif "OrgName:" in line:
                orgname = line.split(":", 1)[-1].strip()
            elif "NetName:" in line:
                netname = line.split(":", 1)[-1].strip()

        return ip, netname, orgname, org

    except Exception as e:
        return ip, "Error", "Error", str(e)
        
        
def print_AS_Info_Suspect_Group():
    
    for ip in SUSPECT_GROUP:
        ip, netname, orgname, org = get_whois_info(ip)
        print(f"{ip} -> NetName: {netname}, OrgName: {orgname}, Organization: {org}")
        
    
def calculate_avg_time_delay(df, id_orig_h, id_resp_h):
    # Input validation
    required_columns = {'ts', 'id.orig_h', 'id.resp_h'}

    # Filter DataFrame
    filtered_df = df[(df['id.orig_h'] == id_orig_h) & (df['id.resp_h'] == id_resp_h)].copy()
    
    # Convert timestamps to datetime
    try:
        filtered_df['ts'] = pd.to_datetime(filtered_df['ts'])
    except Exception as e:
        raise ValueError(f"Failed to convert timestamps: {str(e)}")

    # Sort by timestamp
    filtered_df = filtered_df.sort_values('ts')

    # Calculate time differences in seconds
    time_diffs = filtered_df['ts'].diff().dt.total_seconds().dropna()

    # Calculate and return average time delay
    if len(time_diffs) > 0:
        return time_diffs.mean()
    return None
    
def compute_fft_zeek_log(df, sample_time, hz_list):
    # Ensure timestamps are in datetime format
    df['ts'] = pd.to_datetime(df['ts'])
    
    # Set timestamp as index
    df = df.set_index('ts')
    
    # Resample to count events in each sampling interval
    resample_rule = f'{int(sample_time)}S' if sample_time >= 1 else f'{sample_time*1000}ms'
    df_resampled = df.resample(resample_rule).size()
    
    # Extract the signal (event counts)
    signal = df_resampled.values
    
    # Number of samples
    N = len(signal)
    
    # Sampling frequency (samples per second)
    sampling_rate = 1.0 / sample_time
    
    # Compute FFT
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Keep only positive frequencies
    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf = np.abs(yf)[pos_mask]  # Magnitude of FFT
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(xf, yf)
    for i, hz in enumerate(hz_list):
        if i == 0:
            plt.axvline(x=hz, color='k', linestyle='--', label="Beacon avg delay")
        else:
            plt.axvline(x=hz, color='k', linestyle='--')
    plt.title(f'FFT of Zeek Log Event Counts (Sample Time: {sample_time}s)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return xf, yf
    
def compute_beaconing_scores(df: pd.DataFrame, group_by: List[str] = ["id.orig_h", "id.resp_h"]) -> pd.DataFrame:
    # Define expected columns
    COLUMNS = [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
        "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
        "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
        "tunnel_parents", "label", "detailed_label"
    ]
    
    # Ensure required columns exist
    required_cols = ["ts", "id.orig_h", "id.resp_h", "orig_bytes"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame missing required columns: {set(required_cols) - set(df.columns)}")
    
    # Create a copy to avoid modifying the input DataFrame
    df = df.copy()
    
    # Convert ts to datetime
    df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    
    # Convert orig_bytes to numeric, handling missing values
    df["orig_bytes"] = pd.to_numeric(df["orig_bytes"], errors="coerce").fillna(0)
    
    # Group by specified columns and aggregate timestamps and bytes
    def compute_deltas(group):
        # Sort by timestamp
        group = group.sort_values("ts")
        # Compute time deltas (in seconds)
        deltas = group["ts"].diff().dt.total_seconds().dropna().tolist()
        # Get orig_bytes as list
        bytes_list = group["orig_bytes"].tolist()
        # Count connections
        conn_count = len(group)
        # Compute connection duration (in seconds)
        ts_conn_div = (group["ts"].iloc[-1] - group["ts"].iloc[0]).total_seconds() / 90 if len(group) > 1 else 0
        return pd.Series({
            "deltas": deltas,
            "orig_bytes_list": bytes_list,
            "conn_count": conn_count,
            "tsConnDiv": ts_conn_div
        })
    
    # Apply grouping and compute aggregates, excluding grouping columns
    agg_df = df.groupby(group_by).apply(compute_deltas, include_groups=False).reset_index()
    
    # Merge aggregates back to original DataFrame
    df = df.merge(agg_df, on=group_by, how="left")
    
    # Time-based features
    df["tsLow"] = df["deltas"].apply(lambda x: np.percentile(np.array(x), 20) if x else 0)
    df["tsMid"] = df["deltas"].apply(lambda x: np.percentile(np.array(x), 50) if x else 0)
    df["tsHigh"] = df["deltas"].apply(lambda x: np.percentile(np.array(x), 80) if x else 0)
    df["tsBowleyNum"] = df["tsLow"] + df["tsHigh"] - 2 * df["tsMid"]
    df["tsBowleyDen"] = df["tsHigh"] - df["tsLow"]
    df["tsSkew"] = df[["tsLow", "tsMid", "tsHigh", "tsBowleyNum", "tsBowleyDen"]].apply(
        lambda x: x["tsBowleyNum"] / x["tsBowleyDen"] if x["tsBowleyDen"] != 0 and x["tsMid"] != x["tsLow"] and x["tsMid"] != x["tsHigh"] else 0.0,
        axis=1
    )
    df["tsMadm"] = df["deltas"].apply(
        lambda x: np.median(np.abs(np.array(x) - np.median(np.array(x)))) if x else 0
    )
    
    # Data size features (using orig_bytes)
    df["dsLow"] = df["orig_bytes_list"].apply(lambda x: np.percentile(np.array(x), 20) if x else 0)
    df["dsMid"] = df["orig_bytes_list"].apply(lambda x: np.percentile(np.array(x), 50) if x else 0)
    df["dsHigh"] = df["orig_bytes_list"].apply(lambda x: np.percentile(np.array(x), 80) if x else 0)
    df["dsBowleyNum"] = df["dsLow"] + df["dsHigh"] - 2 * df["dsMid"]
    df["dsBowleyDen"] = df["dsHigh"] - df["dsLow"]
    df["dsSkew"] = df[["dsLow", "dsMid", "dsHigh", "dsBowleyNum", "dsBowleyDen"]].apply(
        lambda x: x["dsBowleyNum"] / x["dsBowleyDen"] if x["dsBowleyDen"] != 0 and x["dsMid"] != x["dsLow"] and x["dsMid"] != x["dsHigh"] else 0.0,
        axis=1
    )
    df["dsMadm"] = df["orig_bytes_list"].apply(
        lambda x: np.median(np.abs(np.array(x) - np.median(np.array(x)))) if x else 0
    )
    
    # Time delta score calculation
    df["tsSkewScore"] = 1.0 - abs(df["tsSkew"])
    df["tsMadmScore"] = 1.0 - (df["tsMadm"] / 30.0)
    df["tsMadmScore"] = df["tsMadmScore"].apply(lambda x: 0 if x < 0 else x)
    df["tsConnCountScore"] = df["conn_count"] / df["tsConnDiv"]
    df["tsConnCountScore"] = df["tsConnCountScore"].apply(lambda x: 1.0 if x > 1.0 else x)
    df["tsScore"] = (((df["tsSkewScore"] + df["tsMadmScore"] + df["tsConnCountScore"]) / 3.0) * 1000) / 1000
    
    # Data size score calculation
    df["dsSkewScore"] = 1.0 - abs(df["dsSkew"])
    df["dsMadmScore"] = 1.0 - (df["dsMadm"] / 128.0)
    df["dsMadmScore"] = df["dsMadmScore"].apply(lambda x: 0 if x < 0 else x)
    df["dsSmallnessScore"] = 1.0 - (df["dsMid"] / 8192.0)
    df["dsSmallnessScore"] = df["dsSmallnessScore"].apply(lambda x: 0 if x < 0 else x)
    df["dsScore"] = (((df["dsSkewScore"] + df["dsMadmScore"] + df["dsSmallnessScore"]) / 3.0) * 1000) / 1000
    
    # Overall score
    df["Score"] = (df["dsScore"] + df["tsScore"]) / 2
    
    # Sort by Score
    df = df.sort_values(by="Score", ascending=False, ignore_index=True)
    
    # Select columns to display (only those in original DataFrame plus new scores)
    display_cols = [col for col in df.columns if col in COLUMNS or col in ["Score", "tsScore", "dsScore"]]
    
    return df[display_cols]    
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_connlog.py /path/to/conn.log")
        sys.exit(1)

    log_path = pathlib.Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: File {log_path} does not exist.")
        sys.exit(1)

    # Pre-process
    df = _read_zeek(log_path)    
    df = _coerce_types(df)
    df = _add_temporal_features(df)
    
    # Create suspect hz list
    hz_list = []
    for res_ip in SUSPECT_GROUP:
    #    time_delay = calculate_avg_time_delay(df, "192.168.100.108", res_ip)   # IOT-23 dataset
       time_delay = calculate_avg_time_delay(df, "10.10.140.58", res_ip)        # Our dataset
       hz_list.append(1/time_delay)
       print(f'{res_ip}: {time_delay} sec')
       
    sample_interval = math.floor(1/ min(hz_list) / 16)

    compute_fft_zeek_log(df, sample_interval, hz_list)
    
    res_df = compute_beaconing_scores(df)
    
    # res_df = res_df[(res_df['id.orig_h'] == "192.168.100.108") & (res_df['id.resp_h'] == "147.231.100.5")]      # IOT-23 dataset
    res_df = res_df[(res_df['id.orig_h'] == "10.10.140.58") & ((res_df['id.resp_h'] == "185.125.190.56") | (res_df['id.resp_h'] == "10.10.140.32"))]      # Our dataset - false positive
    
    res_df = res_df.drop_duplicates(subset='id.resp_h')
    print(res_df.head())