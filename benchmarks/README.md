# Benchmarks

Scripts for reproducing the performance results in the main README and for visualizing the
difference between Standard Isolation Forest and Extended Isolation Forest on synthetic data.

## Directory structure

```
benchmarks/
  eif_spark_benchmark.scala      # Spark benchmark (produces the README results table)
  eif_reference_benchmark.py     # Reference Python EIF (sahandha/eif) comparison
  eif_reference_results.csv      # Pre-computed reference Python results
  download_datasets.py           # Download and convert benchmark datasets
  synthetic/
    eif_synthetic_benchmark.scala  # Generate synthetic data + scores
    eif_synthetic_plots.py         # Plot Standard IF vs Extended IF heatmaps
```

## Datasets

The benchmark datasets are not included in this repository due to their size (~80 MB).
Use the download script to fetch them:

```bash
cd benchmarks
python download_datasets.py
```

This downloads .mat files from [ODDS](http://odds.cs.stonybrook.edu/) and the mulcross
dataset from [OpenML](https://www.openml.org/d/40897), then converts them to CSV format
in the `datasets/` directory.

### Dataset sources

| Dataset | Source | URL |
|---|---|---|
| Annthyroid | ODDS | http://odds.cs.stonybrook.edu/annthyroid-dataset/ |
| Arrhythmia | ODDS | http://odds.cs.stonybrook.edu/arrhythmia-dataset/ |
| Breastw | ODDS | http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/ |
| Cardio | ODDS | http://odds.cs.stonybrook.edu/cardiotocography-dataset/ |
| ForestCover | ODDS | http://odds.cs.stonybrook.edu/forestcovercovertype-dataset/ |
| Http | ODDS | http://odds.cs.stonybrook.edu/http-kddcup99-dataset/ |
| Ionosphere | ODDS | http://odds.cs.stonybrook.edu/ionosphere-dataset/ |
| Mammography | ODDS | http://odds.cs.stonybrook.edu/mammography-dataset/ |
| Mulcross | OpenML | https://www.openml.org/d/40897 |
| Pima | ODDS | http://odds.cs.stonybrook.edu/pima-indians-diabetes-dataset/ |
| Satellite | ODDS | http://odds.cs.stonybrook.edu/satellite-dataset/ |
| Shuttle | ODDS | http://odds.cs.stonybrook.edu/shuttle-dataset/ |
| Smtp | ODDS | http://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/ |

## Running the Spark benchmark

Build the isolation-forest jar, then use `spark-shell`:

```bash
# From the repository root
./gradlew build

# Run the benchmark
cd benchmarks
spark-shell --jars ../isolation-forest/build/libs/isolation-forest_3.5.5_2.13-*.jar
scala> :load eif_spark_benchmark.scala
scala> EIFBenchmark.run(spark)                          // default: 100 trees, 10 iterations
scala> EIFBenchmark.run(spark, numIter = 1)             // quick smoke test
scala> EIFBenchmark.run(spark, saveModelDir = Some("/tmp/eif_models"))  // save models
```

## Running the reference Python benchmark

This validates our Spark results against the reference Python EIF implementation:

```bash
# Clone the reference implementation
git clone https://github.com/sahandha/eif.git /tmp/eif_reference

# Install dependencies
pip install numpy scikit-learn

# Run the benchmark
cd benchmarks
python eif_reference_benchmark.py
python eif_reference_benchmark.py --skip    # resume after interruption
```

The `EIF_REFERENCE_PATH` environment variable can override the default clone location.

## Running the synthetic comparison

Generate and visualize Standard IF vs Extended IF on synthetic 2D datasets:

```bash
cd benchmarks/synthetic

# Step 1: Generate scored CSVs (Spark)
spark-shell --jars ../../isolation-forest/build/libs/isolation-forest_3.5.5_2.13-*.jar
scala> :load eif_synthetic_benchmark.scala
scala> EIFSyntheticBenchmark.run(spark)

# Step 2: Generate plots (Python)
pip install numpy pandas matplotlib seaborn
python eif_synthetic_plots.py
python eif_synthetic_plots.py --formats png    # png only
```

The pre-generated heatmap plots are included in the main README.
