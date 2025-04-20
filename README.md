## 1. Environment setup
```
# Clone / copy this project first, then:
python3 -m venv .venv     # Optional but recommended
source .venv/bin/activate

pip install -r requirements.txt
```
Dependencies:
- Python 3
- pandas, pyarrow, numpy
- xgboost, scikit-learn, joblib, tqdm
- ~8 GB RAM for processing full dataset

## 2. Download dataset
Run `data/download_data.py` to download the "network-malware-detection-connection-analysis" dataset.


## 3  Step 1 – Pre‑process Zeek logs

### IoT‑23 dataset (built‑in)

```bash
python scripts/preprocess.py \
    --input-dir data/network-malware-detection-connection-analysis \
    --recursive \
    --output-fp artifacts/iot23_features.parquet
```

The script will stream each `.log` file, engineer temporal features, and write a compressed **Parquet** file.  
Look for the ✅ message at the end, e.g. `✅ Saved 25,011,015 rows → artifacts/iot23_features.parquet`.

### Your own beacon capture

1. Capture traffic (`tcpdump` or Wireshark) while your beacon **and background noise** run together.
2. Convert to Zeek logs:
   ```bash
   zeek -r capture.pcap        # Produces conn.log, etc.
   ```
3. Move the generated logs under `data/custom_beacon/`.
4. Label them (if not already): edit the `conn.log` and set `label=Malicious`, `detailed-label=C&C-Custom` for the flows where `id.resp_h, id.resp_p` match your beacon.
5. Run preprocessing:
   ```bash
   python scripts/preprocess.py \
       --input-dir data/custom_beacon \
       --recursive \
       --output-fp artifacts/custom_features.parquet
   ```


## 4  Step 2 – Train

```bash
python scripts/train.py \
    --train artifacts/iot23_features.parquet \
    --model-dir artifacts/xgb_baseline
```
 
If running out of memory, change `N` to train on less rows.

## 5  Step 3 – Evaluate on new data

```bash
python scripts/evaluate.py \
    --model-dir artifacts/xgb_baseline \
    --test-data artifacts/custom_features.parquet \
    --output-csv reports/custom_scores.csv
```

* Prints metrics if your test data contains a `label` column.  
* Outputs a CSV with each row’s predicted class (`pred`) and probability (`proba`).  
* Saves metrics to `artifacts/xgb_baseline/eval_metrics.json`.


## 6  Next steps

1. **Memory** – switch XGBoost’s `tree_method` to `hist` and remove the sample cap when you want to train on all 25 M rows.
2. **Feature importance** – load `model.joblib` in a notebook and call `model.get_booster().get_score(importance_type="gain")`.


