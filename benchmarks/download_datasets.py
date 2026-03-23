"""
Download benchmark datasets for the isolation forest benchmarks.

Downloads .mat files from the ODDS repository and the mulcross dataset
from OpenML, then converts everything to CSV format in the datasets/ directory.

Prerequisites:
    pip install scipy numpy requests

Usage:
    cd benchmarks
    python download_datasets.py
"""

import os
import sys
import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# ODDS datasets: (name, mat_url, X_key, y_key)
# URLs are from the ODDS repository (http://odds.cs.stonybrook.edu/)
ODDS_DATASETS = [
    ("annthyroid",  "https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=1",  "X", "y"),
    ("arrhythmia",  "https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1",  "X", "y"),
    ("breastw",     "https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=1",     "X", "y"),
    ("cardio",      "https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1",       "X", "y"),
    ("cover",       "https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=1",        "X", "y"),
    ("http",        "https://www.dropbox.com/s/iy9ucsifal754tp/http.mat?dl=1",         "X", "y"),
    ("ionosphere",  "https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=1",   "X", "y"),
    ("mammography", "https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=1",  "X", "y"),
    ("pima",        "https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=1",         "X", "y"),
    ("satellite",   "https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1",    "X", "y"),
    ("shuttle",     "https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1",       "X", "y"),
    ("smtp",        "https://www.dropbox.com/s/dbv2u4830xri7og/smtp.mat?dl=1",         "X", "y"),
]

# Mulcross dataset from OpenML (ID 40897)
MULCROSS_URL = "https://www.openml.org/data/download/16787460/phpfUae7X"


def download_file(url, dest_path):
    """Download a file from a URL."""
    import requests
    print(f"  Downloading {os.path.basename(dest_path)}...", end=" ", flush=True)
    response = requests.get(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"({len(response.content) / 1024:.0f} KB)")


def mat_to_csv(mat_path, csv_path, x_key="X", y_key="y"):
    """Convert a .mat file to CSV (features + label as last column)."""
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    X = np.array(mat[x_key], dtype=np.float64)
    y = np.array(mat[y_key], dtype=np.float64).ravel()
    data = np.column_stack([X, y])
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10g")
    print(f"  Converted {os.path.basename(mat_path)} -> {os.path.basename(csv_path)} "
          f"({X.shape[0]} rows, {X.shape[1]} features)")


def download_mulcross(datasets_dir):
    """Download and convert the mulcross dataset from OpenML."""
    csv_path = os.path.join(datasets_dir, "mulcross.csv")
    if os.path.exists(csv_path):
        print(f"  mulcross.csv already exists, skipping")
        return

    import requests
    print(f"  Downloading mulcross from OpenML...", end=" ", flush=True)
    response = requests.get(MULCROSS_URL, allow_redirects=True, timeout=120)
    response.raise_for_status()
    print(f"({len(response.content) / 1024:.0f} KB)")

    # Parse ARFF-like format: skip header lines, extract numeric data
    lines = response.text.strip().split("\n")
    data_start = False
    rows = []
    for line in lines:
        if line.strip().upper() == "@DATA":
            data_start = True
            continue
        if data_start and line.strip() and not line.startswith("%"):
            parts = line.strip().split(",")
            # Features are all but the last column; last column is class
            # In mulcross: class "Normal" = 0, class "Anomaly" = 1
            features = [float(p) for p in parts[:-1]]
            label = 1.0 if parts[-1].strip().lower() in ("anomaly", "'anomaly'") else 0.0
            rows.append(features + [label])

    data = np.array(rows)
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10g")
    print(f"  Saved mulcross.csv ({data.shape[0]} rows, {data.shape[1] - 1} features)")


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Check dependencies
    try:
        from scipy.io import loadmat
    except ImportError:
        print("Error: scipy is required. Install with: pip install scipy")
        sys.exit(1)
    try:
        import requests
    except ImportError:
        print("Error: requests is required. Install with: pip install requests")
        sys.exit(1)

    print(f"Downloading datasets to {DATASETS_DIR}/\n")

    # Download and convert ODDS datasets
    for name, url, x_key, y_key in ODDS_DATASETS:
        csv_path = os.path.join(DATASETS_DIR, f"{name}.csv")
        if os.path.exists(csv_path):
            print(f"  {name}.csv already exists, skipping")
            continue

        mat_path = os.path.join(DATASETS_DIR, f"{name}.mat")
        try:
            download_file(url, mat_path)
            mat_to_csv(mat_path, csv_path, x_key, y_key)
        except Exception as e:
            print(f"  Error downloading {name}: {e}")
            print(f"  Please download manually from the ODDS website.")
            continue

    # Download mulcross from OpenML
    try:
        download_mulcross(DATASETS_DIR)
    except Exception as e:
        print(f"  Error downloading mulcross: {e}")
        print(f"  Please download manually from https://www.openml.org/d/40897")

    print(f"\nDone. Datasets saved to {DATASETS_DIR}/")


if __name__ == "__main__":
    main()
