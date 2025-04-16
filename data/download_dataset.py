import os
import shutil
import kagglehub

# Set temp download location to script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["KAGGLEHUB_CACHE_DIR"] = script_dir  # downloads into data/kagglehub/

# Download dataset (to kagglehub/agungpambudi/network-malware-detection-connection-analysis)
download_path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")

# Desired target path
target_path = os.path.join(script_dir, "network-malware-detection-connection-analysis")

# If already exists from a previous run, remove it
if os.path.exists(target_path):
    shutil.rmtree(target_path)

# Move the dataset directory to the top-level data/ folder
shutil.move(download_path, target_path)

# (Optional) Clean up the now-empty kagglehub/agungpambudi folders
kagglehub_root = os.path.join(script_dir, "kagglehub")
try:
    shutil.rmtree(kagglehub_root)
except Exception as e:
    print(f"Cleanup warning: {e}")

print("Dataset ready at:", target_path)
