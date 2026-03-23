"""
Run the reference EIF implementation (sahandha/eif, pure Python) on the exact
same benchmark dataset files used by the Spark benchmark, with matching
hyperparameters (100 trees, 256 samples).

This answers the question: do the gaps vs the EIF paper come from our Spark
implementation (retry policy, etc.) or from dataset differences?

Results are saved incrementally to eif_reference_results.csv so progress is
not lost if the script is interrupted. Datasets are processed smallest-first.

Prerequisites:
    git clone https://github.com/sahandha/eif.git /tmp/eif_reference
    pip install numpy scikit-learn

Usage:
    cd benchmarks
    python eif_reference_benchmark.py           # run all datasets
    python eif_reference_benchmark.py --skip    # skip already-completed datasets in CSV
"""

import sys
import os
import time
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Use the pure-Python reference implementation (no Cython build needed)
# Clone from https://github.com/sahandha/eif into /tmp/eif_reference first
EIF_REFERENCE_PATH = os.environ.get("EIF_REFERENCE_PATH", "/tmp/eif_reference")
sys.path.insert(0, EIF_REFERENCE_PATH)
from eif_old import iForest

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

DATASETS = [
    ("http.csv",        0.004),
    ("cover.csv",       0.009),
    ("mulcross.csv",    0.10),
    ("smtp.csv",        0.003),
    ("shuttle.csv",     0.07),
    ("mammography.csv", 0.0232),
    ("annthyroid.csv",  0.0742),
    ("satellite.csv",   0.32),
    ("pima.csv",        0.35),
    ("breastw.csv",     0.35),
    ("arrhythmia.csv",  0.15),
    ("ionosphere.csv",  0.36),
    ("cardio.csv",      0.096),
]

N_TREES = 100
SAMPLE_SIZE = 256
N_ITER = 10
RESULTS_FILE = "eif_reference_results.csv"


def load_data(path):
    """Load a benchmark CSV. Last column is the label."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split(",")])
    data = np.array(rows)
    return data[:, :-1], data[:, -1]


def evaluate(X, y, extension_level, name, n_iter=N_ITER):
    """Run the reference iForest n_iter times with per-iteration timing."""
    aurocs = []
    auprcs = []
    for seed in range(1, n_iter + 1):
        t0 = time.time()
        np.random.seed(seed)
        import random
        random.seed(seed)
        forest = iForest(X, ntrees=N_TREES, sample_size=SAMPLE_SIZE,
                         ExtensionLevel=extension_level)
        scores = forest.compute_paths(X)
        aurocs.append(roc_auc_score(y, scores))
        auprcs.append(average_precision_score(y, scores))
        elapsed = time.time() - t0
        print(f"    {name} ext={extension_level} iter {seed}/{n_iter}: "
              f"AUROC={aurocs[-1]:.4f} AUPRC={auprcs[-1]:.4f} ({elapsed:.1f}s)",
              flush=True)
    aurocs = np.array(aurocs)
    auprcs = np.array(auprcs)
    return (aurocs.mean(), aurocs.std(ddof=1) / np.sqrt(n_iter),
            auprcs.mean(), auprcs.std(ddof=1) / np.sqrt(n_iter))


def get_completed(results_file):
    """Read already-completed (dataset, ext) pairs from the CSV."""
    completed = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["dataset"], int(row["ext"])))
    return completed


def append_result(results_file, row_dict):
    """Append a single result row to the CSV, creating header if needed."""
    file_exists = os.path.exists(results_file)
    fields = ["dataset", "dim", "ext", "model",
              "auroc_mean", "auroc_sem", "auprc_mean", "auprc_sem"]
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    skip_done = "--skip" in sys.argv

    # Sort datasets by file size (smallest first) for faster early results
    sized = []
    for filename, cont in DATASETS:
        path = os.path.join(DATA_PATH, filename)
        n_rows = sum(1 for line in open(path) if line.strip() and not line.startswith("#"))
        sized.append((n_rows, filename, cont))
    sized.sort()

    completed = get_completed(RESULTS_FILE) if skip_done else set()
    if completed:
        print(f"Skipping {len(completed)} already-completed entries.\n")

    print(f"Reference EIF benchmark: {N_TREES} trees, {SAMPLE_SIZE} samples, "
          f"{N_ITER} iterations\n", flush=True)

    total_start = time.time()

    for n_rows, filename, contamination in sized:
        path = os.path.join(DATA_PATH, filename)
        name = filename.replace(".csv", "")
        X, y = load_data(path)
        dim = X.shape[1]
        max_ext = dim - 1

        # ext=0
        if (name, 0) not in completed:
            print(f"  {name} (n={n_rows}, dim={dim}) ext=0 ...", flush=True)
            m_auc, s_auc, m_apr, s_apr = evaluate(X, y, 0, name)
            append_result(RESULTS_FILE, {
                "dataset": name, "dim": dim, "ext": 0, "model": "IF_ext0",
                "auroc_mean": f"{m_auc:.6f}", "auroc_sem": f"{s_auc:.6f}",
                "auprc_mean": f"{m_apr:.6f}", "auprc_sem": f"{s_apr:.6f}",
            })
            print(f"  -> ext=0: AUROC={m_auc:.4f}+/-{s_auc:.4f}  "
                  f"AUPRC={m_apr:.4f}+/-{s_apr:.4f}", flush=True)

        # ext=max
        if (name, max_ext) not in completed:
            print(f"  {name} (n={n_rows}, dim={dim}) ext={max_ext} ...", flush=True)
            m_auc, s_auc, m_apr, s_apr = evaluate(X, y, max_ext, name)
            append_result(RESULTS_FILE, {
                "dataset": name, "dim": dim, "ext": max_ext, "model": "EIF_max",
                "auroc_mean": f"{m_auc:.6f}", "auroc_sem": f"{s_auc:.6f}",
                "auprc_mean": f"{m_apr:.6f}", "auprc_sem": f"{s_apr:.6f}",
            })
            print(f"  -> ext={max_ext}: AUROC={m_auc:.4f}+/-{s_auc:.4f}  "
                  f"AUPRC={m_apr:.4f}+/-{s_apr:.4f}", flush=True)

        print(flush=True)

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_FILE}")

    # Print summary table
    if os.path.exists(RESULTS_FILE):
        print(f"\n{'Dataset':<15} {'Dim':>3} {'Ext':>3} {'Model':<10} "
              f"{'AUROC':>17} {'AUPRC':>17}")
        print("-" * 75)
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                print(f"{row['dataset']:<15} {row['dim']:>3} {row['ext']:>3} {row['model']:<10} "
                      f"{float(row['auroc_mean']):.4f}+/-{float(row['auroc_sem']):.4f} "
                      f"{float(row['auprc_mean']):.4f}+/-{float(row['auprc_sem']):.4f}")


if __name__ == "__main__":
    main()
