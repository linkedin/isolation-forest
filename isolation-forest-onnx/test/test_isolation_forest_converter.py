import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnxruntime import InferenceSession

from isolationforestonnx.isolation_forest_converter import IsolationForestConverter, _init_tree_ensemble_attrs


BASE_RESOURCES_PATH = Path(__file__).parent / "resources"
ALLOWED_DIFFERENCE = 0.02  # Allowed difference from the expected AUROC.


@pytest.fixture
def converter():
    """
    Fixture to provide an instance of IsolationForestConverter for testing.

    This fixture loads an Isolation Forest model and metadata files from the shuttle dataset.

    :return: An instance of IsolationForestConverter
    """
    # Load the isolation forest model and metadata files.
    model_dir_path = BASE_RESOURCES_PATH / "savedIsolationForestModel" / "shuttleModel"
    model_file_path = list(model_dir_path.glob("data/*.avro"))[0]
    metadata_file_path = model_dir_path / "metadata" / "part-00000"

    return IsolationForestConverter(model_file_path, metadata_file_path)


def _roc_auc_score(y_true, y_score):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from the true labels and the predicted
    scores.

    :param y_true: True labels
    :param y_score: Predicted scores
    :return: ROC AUC
    """

    # Flatten y_score in case it is a 2D array
    y_score = np.ravel(y_score)

    # Convert y_true to numpy array and ensure it is of the correct type
    y_true = np.asarray(y_true, dtype=np.float64)

    # Ensure both classes are present in y_true
    if np.all(y_true == 0) or np.all(y_true == 1):
        raise ValueError("_roc_auc_score cannot be calculated with only one class present in y_true.")

    # Sort the scores and corresponding true labels
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Compute the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)

    # Add the initial point (0, 0) for the ROC curve
    tpr = np.concatenate(([0], tpr))
    fpr = np.concatenate(([0], fpr))

    # Compute the AUC using trapezoidal integration
    auc = np.trapz(tpr, fpr)

    return auc


def _test_converter_on_a_benchmark_dataset(
        dataset_name,
        expected_auroc,
        allowed_difference,
        test_onnx_save_and_load=True) -> None:
    """
    Test the Isolation Forest ONNX converter on a benchmark dataset.

    :param dataset_name: Name of the dataset
    :param expected_auroc: Expected AUROC
    :param allowed_difference: Allowed difference from the expected AUROC
    :param test_onnx_save_and_load: If True, test ONNX model save and load
    :return: None
    """

    input_data = np.loadtxt(BASE_RESOURCES_PATH / f'{dataset_name}.csv', delimiter=',')
    num_features = input_data.shape[1] - 1
    last_col_index = num_features

    # The last column is the label column.
    input_dict = {'features': np.delete(input_data, last_col_index, 1).astype(np.float32)}
    actual_labels = input_data[:, last_col_index]

    # Load the isolation forest model and metadata files.
    model_dir_path = BASE_RESOURCES_PATH / "savedIsolationForestModel" / f'{dataset_name}Model'
    model_file_path = list(model_dir_path.glob("data/*.avro"))[0]
    metadata_file_path = model_dir_path / "metadata" / "part-00000"

    # Convert the isolation forest model to ONNX.
    converter = IsolationForestConverter(model_file_path, metadata_file_path)

    if test_onnx_save_and_load:
        # Test ONNX model convert, save, and load
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
            temp_file_path = temp_file.name
        converter.convert_and_save(temp_file_path)
        onx = onnx.load(temp_file_path)
    else:
        # Test ONNX model convert only
        onx = converter.convert()

    # Create and run an ONNX InferenceSession
    sess = InferenceSession(onx.SerializeToString())
    res = sess.run(None, input_dict)
    predicted_outlier_scores = res[0]

    # Compute the AUROC and compare with the expected AUROC.
    auroc = _roc_auc_score(actual_labels, predicted_outlier_scores)

    assert auroc == pytest.approx(expected_auroc, allowed_difference)


class TestIsolationForestConverter:
    """
    End-to-end tests for the Isolation Forest ONNX converter.

    This class tests the conversion process using various benchmark datasets and verifies that the converted ONNX models
    produce AUROC scores within the allowed difference from expected values.
    """

    @pytest.mark.parametrize("dataset_name, expected_auroc", [
        ('mammography', 0.8596),
        ('shuttle', 0.9976),
    ])
    def test_converter_using_benchmark_datasets(self, dataset_name, expected_auroc):
        """
        Test the ONNX converter using benchmark datasets.

        This test is parameterized with various dataset names and their expected AUROC scores.

        :param dataset_name: Name of the dataset to test
        :param expected_auroc: Expected AUROC score for the dataset
        """
        _test_converter_on_a_benchmark_dataset(dataset_name, expected_auroc, ALLOWED_DIFFERENCE)


def test_get_avg_path_len(converter):
    """
    Test the average path length computation.

    This test verifies the average path length for various numbers of instances.

    :param converter: An instance of IsolationForestConverter
    """
    assert converter._get_avg_path_len(0) == 0.0
    assert converter._get_avg_path_len(1) == 0.0
    assert converter._get_avg_path_len(2) == pytest.approx(0.15443134, rel=1e-5)
    assert converter._get_avg_path_len(10) == pytest.approx(3.7488806, rel=1e-5)
    assert converter._get_avg_path_len(2**63 - 1) == pytest.approx(86.49098, rel=1e-5)


def test_get_depth(converter):
    """
    Test the depth computation of a node in a tree.

    This test verifies the depth of nodes in an isolation forest tree.

    :param converter: An instance of IsolationForestConverter
    """

    # Define a simple tree structure
    parent_ids = {
        1: None,  # Level 1
        2: 1,     # Level 2
        3: 1,     # Level 2
        4: 2,     # Level 3
        5: 2,     # Level 3
        6: 3,     # Level 3
        7: 4,     # Level 4
        8: 4,     # Level 4
        9: 5      # Level 4
    }

    assert converter._get_depth(parent_ids, 1) == 1
    assert converter._get_depth(parent_ids, 2) == 2
    assert converter._get_depth(parent_ids, 3) == 2
    assert converter._get_depth(parent_ids, 4) == 3
    assert converter._get_depth(parent_ids, 5) == 3
    assert converter._get_depth(parent_ids, 6) == 3
    assert converter._get_depth(parent_ids, 7) == 4
    assert converter._get_depth(parent_ids, 8) == 4
    assert converter._get_depth(parent_ids, 9) == 4


def test_add_node_attrs(converter):
    """
    Test the addition of node attributes to the Isolation Forest model.

    This test verifies the correct addition of attributes for nodes, including handling of leaf nodes.

    :param converter: An instance of IsolationForestConverter
    """

    # Mock data
    parent_ids = {}

    # Initialize the attributes dict
    attrs = _init_tree_ensemble_attrs()

    # Mock node data
    node_data = {
        'id': 0,
        'numInstances': 100,
        'leftChild': 1,
        'rightChild': 2,
        'splitAttribute': 1,
        'splitValue': 2.0
    }

    # Call the method to test
    converter._add_node_attrs(tree_id=0, parent_ids=parent_ids, node_data=node_data, attrs=attrs)

    # Asserts for node attributes
    assert attrs['nodes_treeids'] == [0]
    assert attrs['nodes_nodeids'] == [0]
    assert attrs['nodes_featureids'] == [1]
    assert attrs['nodes_modes'] == ['BRANCH_LT']
    assert attrs['nodes_values'] == [2.0]
    assert attrs['nodes_truenodeids'] == [1]
    assert attrs['nodes_falsenodeids'] == [2]
    assert attrs['nodes_missing_value_tracks_true'] == [False]
    assert attrs['nodes_hitrates'] == [1.0]

    # Asserts for target attributes (for leaf nodes)
    assert attrs['target_treeids'] == []
    assert attrs['target_nodeids'] == []
    assert attrs['target_ids'] == []
    assert attrs['target_weights'] == []

    # Check parent IDs
    assert parent_ids == {1: 0, 2: 0}

    # For leaf nodes, ensure target attributes are updated
    node_data_leaf = {
        'id': 1,
        'numInstances': 50,
        'leftChild': -1,
        'rightChild': -1,
        'splitAttribute': -1,
        'splitValue': -1
    }
    converter._add_node_attrs(tree_id=0, parent_ids=parent_ids, node_data=node_data_leaf, attrs=attrs)

    # Verify target attributes are updated
    assert attrs['target_treeids'] == [0]
    assert attrs['target_nodeids'] == [1]
    assert attrs['target_ids'] == [0]
    assert attrs['target_weights'] == [pytest.approx(7.9780719, rel=1e-5)]


def test_add_tree_attrs(converter):
    """
    Test the _add_tree_attrs method of the IsolationForestConverter to ensure it correctly populates the attributes
    dictionary with the expected values when given mock forest data.

    This test mocks a simplified version of an Isolation Forest and checks that the attributes dictionary is correctly
    updated with the expected node and tree attributes.

    :param converter: An instance of IsolationForestConverter
    """

    # Mock the forest data (normally read from the Avro file)
    # This is a simplified mock of nodes in an Isolation Forest.
    converter._forest = [
        {'treeID': 0, 'nodeData': {'id': 0, 'splitAttribute': 1, 'splitValue': 2.0, 'leftChild': 1, 'rightChild': 2, 'numInstances': 100}},
        {'treeID': 0, 'nodeData': {'id': 1, 'splitAttribute': 2, 'splitValue': 3.0, 'leftChild': -1, 'rightChild': -1, 'numInstances': 50}},
        {'treeID': 0, 'nodeData': {'id': 2, 'splitAttribute': 3, 'splitValue': 4.0, 'leftChild': -1, 'rightChild': -1, 'numInstances': 50}},
    ]

    # Initialize the attributes dict
    attrs = _init_tree_ensemble_attrs()

    # Call the method under test
    next_index = converter._add_tree_attrs(tree_id=0, record_index=0, attrs=attrs)

    # Expected values for the tree attributes after adding nodes
    assert attrs['nodes_treeids'] == [0, 0, 0]
    assert attrs['nodes_nodeids'] == [0, 1, 2]
    assert attrs['nodes_featureids'] == [1, 2, 3]
    assert attrs['nodes_modes'] == ['BRANCH_LT', 'LEAF', 'LEAF']
    assert attrs['nodes_values'] == [2.0, 3.0, 4.0]
    assert attrs['nodes_truenodeids'] == [1, -1, -1]
    assert attrs['nodes_falsenodeids'] == [2, -1, -1]
    assert attrs['nodes_missing_value_tracks_true'] == [False, False, False]
    assert attrs['nodes_hitrates'] == [1.0, 1.0, 1.0]

    # Assert the other expected default attributes
    assert attrs['aggregate_function'] == 'AVERAGE'
    assert attrs['n_targets'] == 1
    assert attrs['post_transform'] == 'NONE'

    assert next_index == 3


def test_convert_and_save(converter):
    """
    Test the convert_and_save method of the IsolationForestConverter to ensure it correctly saves the converted ONNX
    model to a file and that the saved model matches the in-memory converted model.

    This test checks two main operations:
    1. Conversion and saving of the ONNX model to a temporary file.
    2. Loading the saved ONNX model from the file and verifying that it matches the model produced by the convert
    method.
    """
    # Test ONNX model convert, save, and load
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
        temp_file_path = temp_file.name
    converter.convert_and_save(temp_file_path)
    onx = onnx.load(temp_file_path)

    assert converter.convert() == onx
