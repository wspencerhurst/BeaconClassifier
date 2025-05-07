# C2 Beacon Detection using Machine Learning

This project implements a machine learning pipeline to detect Command and Control (C2) beaconing activity from Zeek network connection logs. It includes scripts for data preprocessing, feature engineering, model training (XGBoost), evaluation, and visualization. The system is designed to work with the IOT-23 dataset and custom C2 capture data.

## 1. Environment Setup

Ensure you have Python 3.

```bash
# Clone or copy this project, then navigate to the project directory
git clone https://github.com/wspencerhurst/BeaconClassifier.git # Or your repo URL
cd BeaconClassifier

# Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
*   Python 3
*   pandas, pyarrow, numpy
*   xgboost, scikit-learn, joblib, tqdm
*   matplotlib (for `plot_metrics.py`)

**System Requirements:**
*   Processing the full IOT-23 dataset and training can be memory-intensive. At least 8GB RAM is recommended, but I've seen training on the full IOT-23 dataset use over 24GB

## 2. Data Acquisition

### 2.1. IOT-23 Dataset
*   The IOT-23 dataset files (Zeek `conn.log` format, often found as `.log` or `.csv` with pipe delimiters) should be placed in a directory, e.g., `data/iot23_raw/`.
*   If you used the `data/download_data.py` script mentioned previously and it places data in `data/network-malware-detection-connection-analysis/`, ensure your input paths in the commands below reflect this.

### 2.2. Custom C2 Dataset (Your Own Beacon Capture)
1.  **Capture Traffic:** Use tools like `tcpdump` or Wireshark to capture network traffic while your custom C2 beacon is active, preferably with some background noise traffic. Save the capture as a `.pcap` file (e.g., `capture.pcap`).
2.  **Convert to Zeek Logs:**
    ```bash
    zeek -C -r capture.pcap # The -C option ignores checksum errors
    ```
    This will generate several log files, including `conn.log`.
3.  **Organize Logs:** Create a directory (e.g., `data/custom_c2_raw/`) and move the generated `conn.log` file (and any others if needed by different tools, though our classifier only uses `conn.log`) into it. If your logs are from different capture sessions or times and are gzipped (e.g., `conn.18_00_00-19_00_00.log.gz`), place them all in this directory. Our `preprocess.py` script can handle `.gz` files for the `custom` input type.

## 3. Workflow and Commands

The general workflow involves preprocessing raw data, (optionally) preparing a combined training set, training a model, and then evaluating it.

### Step 1: Pre-process Zeek Logs (`scripts/preprocess.py`)

This script converts raw Zeek `conn.log` files into feature-rich Parquet files. It handles different input formats and performs feature engineering (especially temporal IAT features).

**A. Pre-processing the IOT-23 Dataset:**

```bash
python scripts/preprocess.py \
    --input-dir data/iot23_raw \
    --output-fp artifacts/iot23_features.parquet \
    --input-type iot23 \
    --recursive
```
*   `--input-dir`: Directory containing IOT-23 log files.
*   `--output-fp`: Path to save the processed Parquet file.
*   `--input-type iot23`: Specifies the format and labeling logic for IOT-23.
*   `--recursive`: Search for log files in subdirectories.

**B. Pre-processing Your Custom C2 Dataset:**

```bash
python scripts/preprocess.py \
    --input-dir data/custom_c2_raw \
    --output-fp artifacts/custom_c2_features.parquet \
    --input-type custom \
    --victim-ip YOUR_VICTIM_IP \
    --c2-server-ip YOUR_C2_SERVER_IP \
    --recursive
```
*   `--input-dir`: Directory containing your custom C2 `conn.log` (or `conn.*.log.gz`) files.
*   `--input-type custom`: Specifies format and labeling for custom C2.
    *   **Requires:** `--victim-ip` (e.g., `10.10.140.58`) and `--c2-server-ip` (e.g., `10.10.140.32`) to correctly label your C2 traffic as malicious and set the `is_c2` flag.

After running these, you should have:
*   `artifacts/iot23_features.parquet`
*   `artifacts/custom_c2_features.parquet`

### Step 2: Prepare Combined Training Data (`scripts/prepare_combined_dataset.py`)

This script takes the preprocessed IOT-23 data and a portion of your preprocessed custom C2 data to create a combined training set. It also saves a hold-out portion of your custom C2 data for testing the combined model.

```bash
python scripts/prepare_combined_dataset.py \
    --iot23-input artifacts/iot23_features.parquet \
    --custom-input artifacts/custom_c2_features.parquet \
    --combined-train-output artifacts/combined_train_features.parquet \
    --custom-holdout-output artifacts/custom_c2_holdout_test_features.parquet \
    --custom-train-fraction 0.7 \
    --random-seed 42
```
This will create:
*   `artifacts/combined_train_features.parquet` (IOT-23 + 70% of your custom C2 data)
*   `artifacts/custom_c2_holdout_test_features.parquet` (the remaining 30% of your custom C2 data)

### Step 3: Train the XGBoost Model (`scripts/train.py`)

This script trains the XGBoost classifiers (`model_mal` for Malicious/Benign and `model_c2` for C2/Non-C2).

**A. Training Model A (on IOT-23 data only):**

```bash
python scripts/train.py \
    --train artifacts/iot23_features.parquet \
    --model-dir artifacts/model_A_iot23_trained \
    --test-size 0.2 \
    --random-seed 42
```
*   `--train`: Path to the training data Parquet file.
*   `--model-dir`: Directory to save the trained models (`model_mal.joblib`, `model_c2.joblib`) and `feature_columns.joblib`.
*   `--test-size`: Fraction of training data to hold out for internal validation during this training run.

**B. Training Model B (on combined data):**

```bash
python scripts/train.py \
    --train artifacts/combined_train_features.parquet \
    --model-dir artifacts/model_B_combined_trained \
    --test-size 0.2 \
    --random-seed 42
```

### Step 4: Evaluate Models and Plot Metrics (`scripts/evaluate.py`, `scripts/plot_metrics.py`)

These scripts evaluate a trained model on new test data and generate visualizations.

**General Evaluation Command Structure (`evaluate.py`):**

```bash
python scripts/evaluate.py \
    --model-dir path/to/your/model_directory \
    --test-data path/to/your/test_features.parquet \
    --output-csv reports/some_predictions.csv \
    --task c2 # or 'mal' for the Malicious/Benign model
```
*   This prints metrics if the test data has ground truth labels (`label` or `is_c2`).
*   Outputs a CSV with predictions (`pred`) and probabilities (`proba`).
*   Saves detailed metrics to `eval_metrics.json` within the `--model-dir`.

**General Plotting Command Structure (`plot_metrics.py`):**

```bash
python scripts/plot_metrics.py \
    --model-dir path/to/your/model_directory \
    --test-data path/to/your/test_features.parquet \
    --output-dir reports/some_plots_directory \
    --task c2 \
    --sample 100000 # Optional: samples N rows from test data for faster plotting
```
*   Generates `confusion_matrix_thresh_0.5.png`, `roc_curve.png`, `precision_recall_curve.png`, and `feature_importance.png`.
*   Prints metrics for various thresholds to the console.

---

## 4. Interpreting Results

*   **Metrics Files:** `eval_metrics.json` (in each model directory) contains detailed metrics for the default 0.5 threshold (or as evaluated by `evaluate.py`).
*   **Console Output from `plot_metrics.py`:** Provides precision, recall, F1 for various thresholds – crucial for understanding threshold impact.
*   **Plots (`reports/` subdirectories):**
    *   **Confusion Matrix:** Visualizes true/false positives/negatives for a given threshold (default 0.5 in `plot_metrics.py`).
    *   **ROC Curve:** Shows model discrimination ability across all thresholds. Higher AUC is better.
    *   **Precision-Recall Curve:** Illustrates the trade-off between precision and recall. Essential for selecting an operational threshold, especially in security contexts.
    *   **Feature Importance:** Shows which features contributed most to the XGBoost model's predictions (by gain).

## 5. Next Steps / Customization

*   **Feature Engineering:** Modify `preprocess.py` to add or change features.
*   **Model Tuning:** Adjust hyperparameters in `train.py`.
*   **Threshold Selection:** The `plot_metrics.py` output and the precision-recall curve are key. For operational use, you'd select a threshold from the PR curve or based on F1-scores that balances your desired recall and precision.
*   **Memory Management:**
    *   The `train.py` script samples 500,000 rows by default if the dataset is larger. Remove or adjust the `df = df.sample(n=500_000, ...)` line to use more/all data.
    *   XGBoost's `tree_method="hist"` is already used in `train.py` for better memory efficiency.