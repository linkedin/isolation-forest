import os
import numpy as np
import pandas as pd
import pytest
from onnxruntime import InferenceSession
from isolationforestonnx.isolation_forest_converter import IsolationForestConverter

SPARK_VERSION = os.environ.get("SPARK_VERSION")
SCALA_VERSION_SHORT = os.environ.get("SCALA_VERSION_SHORT")


def test_isolation_forest_onnx_integration_end_to_end():
    """
    This is the second part of the end-to-end integration test. The first part is in
    the Spark codebase, where we export the model and score some data. This test
    loads the model and data, converts the model to ONNX, and compares the scores
    between Spark and ONNX.
    """

    base_path = "/tmp/isolationForestModelAndDataForONNX" + "_" + SPARK_VERSION + "_" + SCALA_VERSION_SHORT

    model_path = os.path.join(base_path, "model")
    csv_path = os.path.join(base_path, "scored")

    # 1) data_dir => find the real Avro
    data_dir = os.path.join(model_path, "data")
    avro_file = None
    for f in os.listdir(data_dir):
        if f.startswith("_") or f.startswith("._") or f.endswith(".crc"):
            continue
        if f.endswith(".avro"):
            avro_file = f
            break
    if not avro_file:
        pytest.fail("No .avro found in " + data_dir)
    model_file_path = os.path.join(data_dir, avro_file)

    # 2) meta_dir => find the real JSON
    meta_dir = os.path.join(model_path, "metadata")
    meta_file = None
    for f in os.listdir(meta_dir):
        if f.startswith("_") or f.startswith("._") or f.endswith(".crc"):
            continue
        # Typically "part-00000"
        meta_file = f
        break
    if not meta_file:
        pytest.fail("No metadata file found in " + meta_dir)
    meta_file_path = os.path.join(meta_dir, meta_file)

    # 3) Convert model to ONNX
    print("model_file_path", model_file_path)
    print("meta_file_path", meta_file_path)
    converter = IsolationForestConverter(model_file_path, meta_file_path)
    onnx_model = converter.convert()

    # 4) Load CSV
    if os.path.isdir(csv_path):
        # likely has part-0000
        files = [p for p in os.listdir(csv_path) if p.startswith("part-")]
        if not files:
            pytest.fail("No part files in " + csv_path)
        csv_path = os.path.join(csv_path, files[0])

    df = pd.read_csv(csv_path)
    spark_scores = df["sparkScore"].values

    # We assume columns f0..f5
    feat_cols = [f"f{i}" for i in range(6)]
    if not all(col in df.columns for col in feat_cols):
        pytest.fail("Missing f0..f5 columns in the CSV")

    features_np = df[feat_cols].to_numpy(dtype=np.float32)

    # 5) ONNX inference
    sess = InferenceSession(onnx_model.SerializeToString())
    onnx_scores = sess.run(None, {"features": features_np})[0].flatten()

    # 6) Compare
    diffs = np.abs(spark_scores - onnx_scores)
    max_diff = diffs.max()
    min_diff = diffs.min()
    avg_diff = diffs.mean()
    median_diff = np.median(diffs)
    print(f"Spark vs ONNX: maxDiff={max_diff:.12f}, minDiff={min_diff:.12f} avgDiff={avg_diff:.12f}, medianDiff={median_diff:.12f}")

    assert max_diff < 1e-5, f"Max difference too large! {max_diff}"
