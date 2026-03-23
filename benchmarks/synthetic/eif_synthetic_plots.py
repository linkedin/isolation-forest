"""
Reads the CSVs produced by eif_synthetic_benchmark.scala and generates
side-by-side comparison plots of Standard IF vs Extended IF for each
synthetic dataset.

Plots per dataset:
  1. Scatter plot with top-10 / bottom-10 anomalies highlighted
  2. Score distribution (histogram + KDE) with shared x-axis
  3. Score heatmap (filled contour from meshgrid scores) with shared color scale

Usage:
    python eif_synthetic_plots.py                              # default: png, svg, pdf
    python eif_synthetic_plots.py --formats png                # png only
    python eif_synthetic_plots.py --input-dir eif_synthetic_output --output-dir plots
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_color_codes()

DATASETS = [
    {
        "name": "single_blob",
        "title": "Single Blob",
        "xlim": (-5, 5),
        "ylim": (-5, 5),
        "aspect": "equal",
    },
    {
        "name": "two_blobs",
        "title": "Two Blobs",
        "xlim": (-5, 15),
        "ylim": (-5, 15),
        "aspect": "equal",
    },
    {
        "name": "sinusoid",
        "title": "Sinusoid",
        "xlim": (-5, 30),
        "ylim": (-3, 3),
        "aspect": "auto",
    },
]

MODELS = [
    ("standard_if", "Standard IF"),
    ("extended_if", "Extended IF"),
]


def load_csv(input_dir, dataset_name, model_key, suffix):
    path = os.path.join(input_dir, f"{dataset_name}_{model_key}_{suffix}.csv")
    return pd.read_csv(path)


def infer_grid_size(grid_df):
    """Infer the NxN grid size from the number of rows."""
    n = len(grid_df)
    side = int(np.sqrt(n) + 0.5)
    assert side * side == n, f"Grid has {n} rows, not a perfect square"
    return side


def plot_scatter(ax, data, title, xlim, ylim, aspect):
    """Scatter plot with top-10 (black) and bottom-10 (red) anomaly scores."""
    scores = data["outlierScore"].values
    sorted_idx = np.argsort(scores)

    ax.scatter(data["x"], data["y"], s=15, c="b", edgecolor="b", alpha=0.6)
    top10 = sorted_idx[-10:]
    ax.scatter(data["x"].iloc[top10], data["y"].iloc[top10], s=55, c="k",
               zorder=5, label="Top 10 (most anomalous)")
    bot10 = sorted_idx[:10]
    ax.scatter(data["x"].iloc[bot10], data["y"].iloc[bot10], s=55, c="r",
               zorder=5, label="Top 10 (least anomalous)")

    ax.set_title(title, fontsize=13)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)
    ax.legend(fontsize=7, loc="upper right")


def plot_score_distribution(ax, data, title):
    """Histogram + KDE of anomaly scores."""
    sns.histplot(data["outlierScore"], kde=True, color="b", ax=ax, stat="density")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Anomaly Score")


def plot_heatmap(ax, grid, data, title, xlim, ylim, aspect, vmin, vmax):
    """Filled contour plot from meshgrid scores with shared color scale."""
    n_grid = infer_grid_size(grid)
    xx = grid["x"].values.reshape(n_grid, n_grid)
    yy = grid["y"].values.reshape(n_grid, n_grid)
    zz = grid["outlierScore"].values.reshape(n_grid, n_grid)

    levels = np.linspace(vmin, vmax, 15)
    cs = ax.contourf(xx, yy, zz, levels=levels, cmap=plt.cm.YlOrRd)
    ax.scatter(data["x"], data["y"], s=3, c="k", alpha=0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(aspect)
    return cs


def save_fig(fig, output_dir, name, formats):
    """Save a figure in all requested formats."""
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  {name}.{fmt}")


def generate_plots(input_dir, output_dir, formats):
    os.makedirs(output_dir, exist_ok=True)

    for ds in DATASETS:
        name = ds["name"]
        title = ds["title"]
        xlim = ds["xlim"]
        ylim = ds["ylim"]
        aspect = ds["aspect"]

        # Load all data
        data = {}
        grids = {}
        for model_key, _ in MODELS:
            data[model_key] = load_csv(input_dir, name, model_key, "data")
            grids[model_key] = load_csv(input_dir, name, model_key, "grid")

        # --- 1. Scatter plots ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (model_key, model_label) in zip(axes, MODELS):
            plot_scatter(ax, data[model_key], model_label, xlim, ylim, aspect)
        fig.suptitle(f"{title}: Top Anomalies", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_fig(fig, output_dir, f"{name}_scatter", formats)
        plt.close(fig)

        # --- 2. Score distributions (shared x-axis range) ---
        all_scores = np.concatenate([data[mk]["outlierScore"].values
                                     for mk, _ in MODELS])
        score_min = all_scores.min()
        score_max = all_scores.max()
        pad = (score_max - score_min) * 0.05

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (model_key, model_label) in zip(axes, MODELS):
            plot_score_distribution(ax, data[model_key], model_label)
            ax.set_xlim(score_min - pad, score_max + pad)
        fig.suptitle(f"{title}: Score Distributions", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_fig(fig, output_dir, f"{name}_score_dist", formats)
        plt.close(fig)

        # --- 3. Score heatmaps (shared color scale + colorbar) ---
        all_grid_scores = np.concatenate([grids[mk]["outlierScore"].values
                                          for mk, _ in MODELS])
        vmin = all_grid_scores.min()
        vmax = all_grid_scores.max()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        cs = None
        for ax, (model_key, model_label) in zip(axes, MODELS):
            cs = plot_heatmap(ax, grids[model_key], data[model_key],
                              model_label, xlim, ylim, aspect, vmin, vmax)
        fig.suptitle(f"{title}: Score Maps", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.92, 0.95])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        fig.colorbar(cs, cax=cbar_ax, label="Anomaly Score")
        save_fig(fig, output_dir, f"{name}_heatmap", formats)
        plt.close(fig)

    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IF vs EIF comparison plots from scored CSVs.")
    parser.add_argument("--input-dir", default="eif_synthetic_output",
                        help="Directory containing CSVs from the Scala benchmark")
    parser.add_argument("--output-dir", default="eif_synthetic_plots",
                        help="Directory to write plot files")
    parser.add_argument("--formats", default="png,svg,pdf",
                        help="Comma-separated output formats (default: png,svg,pdf)")
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",")]
    generate_plots(args.input_dir, args.output_dir, formats)


if __name__ == "__main__":
    main()
