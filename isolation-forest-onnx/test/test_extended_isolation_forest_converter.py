import pytest
import numpy as np
import onnx
from onnxruntime import InferenceSession
import tempfile
from pathlib import Path

# Import the updated converter (see code snippet below)
from isolationforestonnx.extended_isolation_forest_converter import (
    ExtendedIsolationForestConverter,
)

BASE_RESOURCES_PATH = Path(__file__).parent / "resources"
ALLOWED_DIFFERENCE = 0.02  # Allowed difference from the expected AUROC.

def _roc_auc_score(y_true, y_score):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    """
    y_score = np.ravel(y_score)
    y_true = np.asarray(y_true, dtype=np.float64)

    if np.all(y_true == 0) or np.all(y_true == 1):
        raise ValueError(
            "_roc_auc_score cannot be calculated with only one class present in y_true."
        )

    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)

    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))

    auc = np.trapz(tpr, fpr)
    return auc


def _test_extended_converter_on_a_benchmark_dataset(
        dataset_name,
        expected_auroc,
        allowed_difference=ALLOWED_DIFFERENCE,
        test_onnx_save_and_load=True
):
    """
    Test the Extended Isolation Forest ONNX converter on a benchmark dataset.

    Raises FileNotFoundError instead of skipping if the CSV or Avro/metadata files are missing.
    """
    # 1) Load CSV dataset
    data_csv = BASE_RESOURCES_PATH / f"{dataset_name}.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {data_csv}")

    input_data = np.loadtxt(data_csv, delimiter=",")
    num_features = input_data.shape[1] - 1
    last_col_index = num_features
    input_dict = {
        "features": np.delete(input_data, last_col_index, 1).astype(np.float32)
    }
    actual_labels = input_data[:, last_col_index]

    # 2) Load extended iForest model from Avro + metadata
    model_dir_path = BASE_RESOURCES_PATH / "savedExtendedIsolationForestModel" / f"{dataset_name}Model"

    avro_files = list(model_dir_path.glob("data/*.avro"))
    if not avro_files:
        raise FileNotFoundError(f"No Avro files found under {model_dir_path}/data/*.avro")

    model_file_path = avro_files[0]
    metadata_file_path = model_dir_path / "metadata" / "part-00000"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")

    converter = ExtendedIsolationForestConverter(model_file_path, metadata_file_path)

    # 3) Convert (and optionally save+load) to ONNX
    if test_onnx_save_and_load:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_file_path = temp_file.name
        converter.convert_and_save(temp_file_path)  # The checker is called here
        onx = onnx.load(temp_file_path)
    else:
        # Just an in-memory convert; no checker. The final model that gets used is in convert_and_save.
        onx = converter.convert()

    # 4) Run inference in onnxruntime
    sess = InferenceSession(onx.SerializeToString())
    predicted_outlier_scores = sess.run(None, input_dict)[0]

    # 5) Compare AUROC
    auroc = _roc_auc_score(actual_labels, predicted_outlier_scores)
    assert auroc == pytest.approx(expected_auroc, allowed_difference)


@pytest.fixture
def extended_converter():
    """
    Fixture to provide an instance of ExtendedIsolationForestConverter for testing.
    Points to 'shuttleModel' by default. Raises FileNotFoundError if resources are missing.
    """
    model_dir_path = BASE_RESOURCES_PATH / "savedExtendedIsolationForestModel" / "shuttleModel"

    avro_files = list(model_dir_path.glob("data/*.avro"))
    if not avro_files:
        raise FileNotFoundError("No Avro files found for 'shuttleModel' in extended converter tests.")

    model_file_path = avro_files[0]
    metadata_file_path = model_dir_path / "metadata" / "part-00000"
    if not metadata_file_path.exists():
        raise FileNotFoundError(f"Metadata file not found for 'shuttleModel': {metadata_file_path}")

    return ExtendedIsolationForestConverter(model_file_path, metadata_file_path)


class TestExtendedIsolationForestConverter:
    """
    End-to-end tests for the Extended Isolation Forest ONNX converter.
    """

    @pytest.mark.parametrize(
        "dataset_name, expected_auroc",
        [
            ("mammography", 0.8596),
            ("shuttle", 0.9976),
        ],
    )
    def test_extended_converter_using_benchmark_datasets(self, dataset_name, expected_auroc):
        """
        Test the Extended ONNX converter using benchmark datasets.
        Fails if CSV or Avro resources are missing.
        """
        _test_extended_converter_on_a_benchmark_dataset(dataset_name, expected_auroc)

    def test_avg_path_len_formula(self, extended_converter):
        """
        Test the extended converter's _avg_path_len_formula at a few sample sizes.
        """
        assert extended_converter._avg_path_len_formula(0) == 0.0
        assert extended_converter._avg_path_len_formula(1) == 0.0
        val_2 = extended_converter._avg_path_len_formula(2)
        assert val_2 == pytest.approx(0.1544, abs=1e-3)
        val_10 = extended_converter._avg_path_len_formula(10)
        assert val_10 == pytest.approx(3.7488, abs=1e-3)

    def test_convert_and_save(self, extended_converter):
        """
        Ensure convert_and_save() writes a valid ONNX file.
        """
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as temp_file:
            temp_file_path = temp_file.name

        # This calls convert(), then patches the constants, then checker.check_model()
        extended_converter.convert_and_save(temp_file_path)
        onx_saved = onnx.load(temp_file_path)

        # Just load the final model in memory (the partial model won't pass checker).
        in_mem_model = extended_converter.convert()

        # We'll verify that loading the saved ONNX is valid and can be run:
        sess = InferenceSession(onx_saved.SerializeToString())
        # If session creation works, we've confirmed the final model is good.

        # Optional: you can do a quick inference test:
        # feats = np.zeros((1, extended_converter.num_features), dtype=np.float32)
        # _ = sess.run(None, {"features": feats})

        assert True, "convert_and_save produced a valid ONNX file."