import os
import numpy as np
import pandas as pd
import pytest
import tempfile
import onnx
from onnxruntime import InferenceSession

# Import the EXTENDED converter
from isolationforestonnx.extended_isolation_forest_converter import ExtendedIsolationForestConverter

SPARK_VERSION = os.environ.get("SPARK_VERSION")
SCALA_VERSION_SHORT = os.environ.get("SCALA_VERSION_SHORT")


def _run_batch_single_row(sess, x: np.ndarray):
    """
    Runs inference row-by-row on a single-row ONNX model.
    x is shape [N, num_features].
    Returns outlier_scores and predicted_labels as arrays of length N.
    """
    n = x.shape[0]
    out_scores = []
    out_labels = []
    for i in range(n):
        row_input = x[i:i+1, :]  # shape [1, num_features]
        # The model returns outlier_score, predicted_label both shape [1,1]
        outlier_score, predicted_label = sess.run(
            None,
            {"features": row_input.astype(np.float32)}
        )
        # Flatten them from shape [1,1] => scalar
        out_scores.append(outlier_score[0, 0])
        out_labels.append(predicted_label[0, 0])

    return np.array(out_scores), np.array(out_labels)


def test_extended_isolation_forest_onnx_integration_end_to_end():
    """
    Integration test for the extended isolation forest ONNX converter,
    mirroring the style of the standard isolation forest test.

    We assume that Spark has already exported:
      1) The extended iForest Avro (model.avro)
      2) The metadata JSON
      3) A CSV with columns f0...fN plus 'sparkScore'
    under /tmp/extendedIsolationForestModelAndDataForONNX_<sparkVer>_<scalaVer>.
    """

    # 1) Build the base path that Spark wrote
    base_path = (
            "/tmp/extendedIsolationForestModelAndDataForONNX"
            + "_" + SPARK_VERSION
            + "_" + SCALA_VERSION_SHORT
    )

    model_path = os.path.join(base_path, "model")
    csv_path = os.path.join(base_path, "scored")

    # 2) Find the Avro model file
    data_dir = os.path.join(model_path, "data")
    avro_file = None
    for f in os.listdir(data_dir):
        if f.startswith("_") or f.startswith("._") or f.endswith(".crc"):
            continue
        if f.endswith(".avro"):
            avro_file = f
            break
    if not avro_file:
        pytest.fail("No .avro file found in " + data_dir)
    model_file_path = os.path.join(data_dir, avro_file)

    # 3) Find the metadata JSON
    meta_dir = os.path.join(model_path, "metadata")
    meta_file = None
    for f in os.listdir(meta_dir):
        if f.startswith("_") or f.startswith("._") or f.endswith(".crc"):
            continue
        meta_file = f
        break
    if not meta_file:
        pytest.fail("No metadata file found in " + meta_dir)
    meta_file_path = os.path.join(meta_dir, meta_file)

    print("Extended model_file_path =", model_file_path)
    print("Extended meta_file_path  =", meta_file_path)

    # 4) Convert -> Save to a temp .onnx file. This ensures the final model
    #    includes the new constant nodes that define tree_{i}_tripCount_out, etc.
    converter = ExtendedIsolationForestConverter(model_file_path, meta_file_path)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
        temp_model_path = temp_file.name
    converter.convert_and_save(temp_model_path)

    # 5) Load the final ONNX from disk
    onx_model = onnx.load(temp_model_path)
    sess = InferenceSession(onx_model.SerializeToString())

    # 6) Load the CSV that has columns f0..fN plus 'sparkScore'
    if os.path.isdir(csv_path):
        files = [p for p in os.listdir(csv_path) if p.startswith("part-")]
        if not files:
            pytest.fail("No part files in " + csv_path)
        csv_path = os.path.join(csv_path, files[0])

    df = pd.read_csv(csv_path)
    if "sparkScore" not in df.columns:
        pytest.fail("CSV missing 'sparkScore' column.")
    spark_scores = df["sparkScore"].values

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        pytest.fail("No feature columns named f0.. in the CSV.")
    feat_cols.sort()

    features_np = df[feat_cols].to_numpy(dtype=np.float32)

    # 7) ONNX inference row-by-row for single-row model
    onnx_scores, _ = _run_batch_single_row(sess, features_np)

    # 8) Compare Spark vs. ONNX
    diffs = np.abs(spark_scores - onnx_scores)
    max_diff = diffs.max()
    print("Extended iForest max difference in sparkScore vs ONNX:", max_diff)
    assert max_diff < 1e-5, f"Max difference too large for Extended iForest! {max_diff}"
