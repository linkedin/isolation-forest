import json
import logging

from typing import Any, Dict, List, cast

import numpy as np
import onnx
from avro.datafile import DataFileReader
from avro.io import DatumReader
from onnx import ModelProto, TensorProto, NodeProto, checker, helper

logger = logging.getLogger(__name__)


def _init_tree_ensemble_attrs() -> Dict[str, Any]:
    """
    Initialize the attributes for the TreeEnsembleRegressor node.

    :return: Dictionary of attributes
    """
    attrs = {
        'aggregate_function': 'AVERAGE',
        'n_targets': 1,
        'nodes_falsenodeids': [],
        'nodes_featureids': [],
        'nodes_hitrates': [],
        'nodes_missing_value_tracks_true': [],
        'nodes_modes': [],
        'nodes_nodeids': [],
        'nodes_treeids': [],
        'nodes_truenodeids': [],
        'nodes_values': [],
        'post_transform': 'NONE',
        'target_ids': [],
        'target_nodeids': [],
        'target_treeids': [],
        'target_weights': []
    }
    return attrs


class IsolationForestConverter:
    """
    Converts an Isolation Forest model to an ONNX model.

    The ONNX graph computes the outlier score and label for the Isolation Forest algorithm.

    :param model_file_path: Path to the saved Isolation Forest Avro file
    :param metadata_file_path: Path to the saved Isolation Forest model metadata JSON file
    """
    def __init__(self, model_file_path: str, metadata_file_path: str):
        # Get model hyper-parameters and other metadata
        try:
            with open(metadata_file_path, 'r') as file:
                metadata = json.load(file)
                params = metadata['paramMap']
                self._num_trees = params['numEstimators']
                self._outlier_score_threshold = metadata['outlierScoreThreshold']
                self._num_samples = metadata['numSamples']
                self._max_features = params['maxFeatures']
                self._contamination = params['contamination']
                self._max_samples = params['maxSamples']
                self._num_features = metadata['numFeatures']

            logger.info(f'Isolation Forest model metadata: num_trees: {self._num_trees}, '
                        f'max_features: {self._max_features}, contamination: {self._contamination}, '
                        f'max_samples: {self._max_samples}, num_features: {self._num_features}')
        except Exception as e:
            logger.error(f'Error loading metadata: {e}')
            raise

        # Read the Isolation Forest Avro model file
        try:
            with open(model_file_path, 'rb') as in_stream:
                with DataFileReader(in_stream, DatumReader()) as reader:
                    self._forest = [node for node in reader]
                    logger.info(f'Read the Isolation Forest model file; it has {len(self._forest)} tree nodes')
                    # Log the first few nodes
                    logger.info('First two tree nodes:')
                    for idx in range(min(2, len(self._forest))):
                        logger.info(json.dumps(self._forest[idx], indent=2))

            self._forest.sort(key=self.get_tree_id)
        except Exception as e:
            logger.error(f'Error reading Avro file: {e}')
            raise

    def get_tree_id(self, node: Dict[str, Any]) -> int:
        """
        Returns the tree ID of a node.

        :param node: Node in the Isolation Forest model
        :return: Tree ID
        """
        return int(node['treeID'])

    def convert_and_save(self, onnx_model_path: str) -> None:
        """
        Converts the model to ONNX representation and saves it.

        :param onnx_model_path: Path to save the ONNX model
        """
        try:
            model_proto = self.convert()
            onnx.save_model(model_proto, onnx_model_path)
        except Exception as e:
            logger.error(f'Error converting and saving ONNX model: {e}')
            raise

    def convert(self) -> ModelProto:
        """
        Converts the model to ONNX representation.

        :return: ONNX ModelProto object
        """
        features = helper.make_tensor_value_info('features', TensorProto.FLOAT, [None, self._num_features])
        nodes: List[NodeProto] = [
            self._create_tree_ensemble_regressor(),
            self._create_avg_path_len(),
            self._create_path_len_normalizer(),
            self._create_normalized_path_len_neg(),
            self._create_constant_2(),
            self._create_constant_outlier_score_threshold(),
            self._create_outlier_score(),
            self._create_less(),
            self._create_predicted_label_boolean(),
            self._create_predicted_label()
        ]

        outlier_score = helper.make_tensor_value_info('outlier_score', TensorProto.FLOAT, [None, 1])
        predicted_label = helper.make_tensor_value_info('predicted_label', TensorProto.INT32, [None, 1])

        graph = helper.make_graph(
            nodes=nodes,
            name='IsolationForestGraph',
            inputs=[features],
            outputs=[outlier_score, predicted_label]
        )

        model = helper.make_model(
            graph=graph,
            producer_name='IsolationForestSparkMlToOnnxConverter',
            opset_imports=[
                helper.make_opsetid('ai.onnx.ml', 1),
                helper.make_opsetid('', 14)
            ]
        )

        try:
            checker.check_model(model)
            logger.info('Successfully converted to ONNX model')
        except checker.ValidationError as e:
            logger.error(f'The converted ONNX model is invalid: {e}')
            raise

        return model

    def _create_tree_ensemble_regressor(self) -> NodeProto:
        """
        Create the TreeEnsembleRegressor node that computes the expected path length E[h(x)] of a sample x in an
        Isolation Forest model.

        :return: NodeProto object
        """
        attrs = _init_tree_ensemble_attrs()

        # Add the nodes and target attributes for each tree in the forest
        next_tree_index = 0
        for tree_id in range(self._num_trees):
            next_tree_index = self._add_tree_attrs(tree_id, next_tree_index, attrs)

        return helper.make_node(
            op_type='TreeEnsembleRegressor',
            inputs=['features'],
            outputs=['expected_path_len'],
            name='TreeEnsembleRegressorNode',
            doc_string='This node computes the expected path length E[h(x)] of a sample x in an Isolation Forest model',
            domain='ai.onnx.ml',
            **attrs
        )

    def _create_avg_path_len(self) -> NodeProto:
        """
        Create a node that computes the average path length c(n) of an unsuccessful search in a Binary Search Tree.

        :return: NodeProto object
        """
        avg_path_len = self._get_avg_path_len(self._num_samples)
        avg_path_len_tensor = helper.make_tensor(
            name='avg_path_len_tensor', data_type=TensorProto.FLOAT, dims=[], vals=[avg_path_len]
        )
        return helper.make_node(
            'Constant', inputs=[], outputs=['avg_path_len'], value=avg_path_len_tensor
        )

    def _create_path_len_normalizer(self) -> NodeProto:
        """
        Create a node that normalizes the expected path length by computing E[h(x)] / c(n).

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Div',
            inputs=['expected_path_len', 'avg_path_len'],  # TODO: Can we get these from the nodeproto?
            outputs=['normalized_path_len'],
            name='PathLenNormalizer',
            doc_string='This node normalizes the expected path length by computing E[h(x)] / c(n)',
        )

    def _create_normalized_path_len_neg(self) -> NodeProto:
        """
        Create a node that negates the normalized path length thereby computing -(E[h(x)] / c(n)).

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Neg',
            inputs=['normalized_path_len'],
            outputs=['neg_normalized_path_len'],
            name='NormalizedPathLenNeg',
            doc_string='This node negates the normalized path length thereby computing -(E[h(x)] / c(n))'
        )

    def _create_constant_2(self) -> ModelProto:
        """
        Create a constant node with value 2.

        :return: NodeProto object
        """
        return helper.make_node(
            'Constant',
            inputs=[],
            outputs=['constant_2'],
            value=helper.make_tensor(
                name='constant_2',
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[2.]
            )
        )

    def _create_outlier_score(self) -> NodeProto:
        """
        Create a node that computes the final anomaly score s(x, n) = 2 ^ -(E[h(x)] / c(n)).

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Pow',
            inputs=['constant_2', 'neg_normalized_path_len'],
            outputs=['outlier_score'],
            name='ScoreNode',
            doc_string='This node computes the final anomaly score s(x, n) = 2 ^ -(E[h(x)] / c(n))'
        )

    def _create_constant_outlier_score_threshold(self) -> NodeProto:
        """
        Create a constant node with the outlier score threshold.

        :return: NodeProto object
        """
        return helper.make_node(
            'Constant',
            inputs=[],
            outputs=['constant_outlier_score_threshold'],
            value=helper.make_tensor(
                name='constant_outlier_score_threshold',
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[self._outlier_score_threshold]
            )
        )

    def _create_less(self) -> NodeProto:
        """
        Create a node that compares the score with the outlier score threshold. If score >= outlierScoreThreshold,
        then output True else output False.

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Less',
            inputs=['outlier_score', 'constant_outlier_score_threshold'],
            outputs=['less'],
            name='ScoreLessThanOutlierScoreThreshold'
        )

    def _create_predicted_label_boolean(self) -> NodeProto:
        """
        Create a node that computes the predicted label as 1 if score >= outlierScoreThreshold else 0.

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Not',
            inputs=['less'],
            outputs=['not'],
            name='IsOutlier'
        )

    def _create_predicted_label(self) -> NodeProto:
        """
        Create a node that casts the boolean predicted label to an integer.

        :return: NodeProto object
        """
        return helper.make_node(
            op_type='Cast',
            inputs=['not'],
            outputs=['predicted_label'],
            to=TensorProto.INT32,
            name='PredictedLabel'
        )

    def _get_avg_path_len(self, num_instances: int) -> float:
        """
        Compute the average path length of an unsuccessful search in a Binary Search Tree.

        :param num_instances: Number of instances used to train a tree
        :return: Average path length
        """
        return 0.0 if num_instances <= 1 else cast(float, 2 * (np.log(num_instances - 1) + np.euler_gamma) - (2 * (num_instances - 1) / num_instances))

    def _get_depth(self, parent_ids: Dict[int, int], node_id: int) -> int:
        """
        Compute the depth of a node in a tree.

        :param parent_ids: Dictionary of parent node IDs
        :param node_id: Node ID
        :return: Depth of the node
        """
        depth = 0
        while node_id in parent_ids:
            depth += 1
            node_id = parent_ids[node_id]
        return depth

    def _add_tree_attrs(self, tree_id: int, record_index: int, attrs: Dict[str, Any]) -> int:
        """
        Add the attributes for each tree to in the Isolation Forest model.

        :param tree_id: Tree ID
        :param record_index: Index of the record in the Avro file
        :param attrs: Dictionary of attributes
        :return: Index of the next record in the Avro file
        """

        # The nodes of each tree in the Isolation Forest model are stored in pre-order traversal.
        # That allows us to determine the parent of each node which is stored in this dict. There's
        # no parent of root, and therefore the entry for root isn't stored.
        parent_ids: Dict[int, int] = {}

        while record_index < len(self._forest) and self._forest[record_index]['treeID'] == tree_id:
            # Each record (obtained from the Avro file) is a node in a tree
            node = self._forest[record_index]
            node_data = node['nodeData']

            self._add_node_attrs(tree_id, parent_ids, node_data, attrs)
            record_index += 1

        return record_index

    def _add_node_attrs(self, tree_id: int, parent_ids: Dict[int, int], node_data: Dict[str, int], attrs: Dict[str, Any]) -> None:
        """
        Add the attributes for each node in the Isolation Forest model.

        :param tree_id: Tree ID
        :param parent_ids: Dictionary of parent node IDs
        :param node_data: Node data
        :param attrs: Dictionary of attributes
        :return: None
        """

        node_id = node_data['id']
        num_instances = node_data['numInstances']
        left_child_id = node_data['leftChild']
        right_child_id = node_data['rightChild']
        parent_ids[left_child_id] = node_id
        parent_ids[right_child_id] = node_id

        is_leaf = left_child_id == -1 and right_child_id == -1
        mode = 'LEAF' if is_leaf else 'BRANCH_LT'
        feature_id = node_data['splitAttribute']
        threshold = node_data['splitValue']

        # See https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md#aionnxmltreeensembleregressor
        attrs['nodes_treeids'].append(tree_id)
        attrs['nodes_nodeids'].append(node_id)
        attrs['nodes_featureids'].append(feature_id)
        attrs['nodes_modes'].append(mode)
        attrs['nodes_values'].append(threshold)
        attrs['nodes_truenodeids'].append(left_child_id)
        attrs['nodes_falsenodeids'].append(right_child_id)
        attrs['nodes_missing_value_tracks_true'].append(False)
        attrs['nodes_hitrates'].append(1.)

        if is_leaf:
            # For a leaf the path length is its depth + an adjustment
            path_len = self._get_depth(parent_ids, node_id) + self._get_avg_path_len(num_instances)
            attrs['target_treeids'].append(tree_id)
            attrs['target_nodeids'].append(node_id)
            attrs['target_ids'].append(0)
            attrs['target_weights'].append(path_len)
