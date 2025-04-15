import kagglehub

# Download latest version
path = kagglehub.dataset_download("agungpambudi/network-malware-detection-connection-analysis")

print("Path to dataset files:", path)