import json
import logging
from typing import Any, Dict, List

import numpy as np
import onnx
from onnx import (
    ModelProto,
    GraphProto,
    NodeProto,
    TensorProto,
    helper,
    checker
)

logger = logging.getLogger(__name__)


class ExtendedIsolationForestConverter:
    """
    Extended Isolation Forest in ONNX, single-row only.

    Key fix: For norms, we only squeeze axis=0 so final shape is [1], not [].
    That way, Concat(axis=0) forms a [nFeatures] vector rather than rank-0 scalars.
    """

    def __init__(self, model_file_path: str, metadata_file_path: str):
        from avro.datafile import DataFileReader
        from avro.io import DatumReader

        # 1) Load metadata (JSON)
        with open(metadata_file_path, 'r') as f:
            meta = json.load(f)
            self.param_map = meta['paramMap']
            self.num_trees = self.param_map['numEstimators']
            self.num_features = meta['numFeatures']
            self.num_samples = meta['numSamples']
            self.outlier_score_threshold = meta['outlierScoreThreshold']

        # 2) Read Avro => self.forest_records
        with open(model_file_path, 'rb') as in_stream:
            reader = DataFileReader(in_stream, DatumReader())
            self.forest_records = [r for r in reader]
            reader.close()

        # 3) Group nodes by treeID and sort by node ID
        by_tree: Dict[int, List[Dict[str, Any]]] = {}
        for record in self.forest_records:
            t_id = record['treeID']
            nd = record['extendedNodeData']
            by_tree.setdefault(t_id, []).append(nd)

        self.trees_data = []
        for t_id in range(self.num_trees):
            arr = sorted(by_tree[t_id], key=lambda x: x['id'])
            self.trees_data.append(arr)

        logger.info(
            f"Loaded extended IF with {self.num_trees} trees. "
            f"Each tree's node array is in self.trees_data."
        )

    def convert(self) -> ModelProto:
        # Main input
        features_info = helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, [None, self.num_features]
        )

        main_nodes: List[NodeProto] = []
        tree_outputs = []

        # 1) Build a loop for each tree
        for i in range(self.num_trees):
            node_table_const_name = f"tree_{i}_nodes"
            self._append_tree_table_constant(i, main_nodes, node_table_const_name)

            path_len_output_name = f"tree_{i}_pathLen"
            loop_node = self._make_loop_node_for_tree(
                i, node_table_const_name, path_len_output_name
            )
            main_nodes.append(loop_node)
            tree_outputs.append(path_len_output_name)

        # 2) Sum path lengths
        if len(tree_outputs) > 1:
            sum_path_name = "sum_path_len"
            main_nodes.append(helper.make_node(
                "Sum",
                inputs=tree_outputs,
                outputs=[sum_path_name]
            ))
        else:
            sum_path_name = tree_outputs[0]

        # 3) average => expected_path_len
        expected_path_len = "expected_path_len"
        denom_const_name = "trees_count_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[denom_const_name],
            value=helper.make_tensor(
                name=denom_const_name + "_val",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(self.num_trees)]
            )
        ))
        main_nodes.append(helper.make_node(
            "Div",
            inputs=[sum_path_name, denom_const_name],
            outputs=[expected_path_len]
        ))

        # 4) normalize by c(n)
        c_n_val = self._avg_path_len_formula(self.num_samples)
        cn_const_name = "cn_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[cn_const_name],
            value=helper.make_tensor(
                name=cn_const_name + "_val",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[c_n_val]
            )
        ))
        norm_pl = "normalized_path_len"
        main_nodes.append(helper.make_node(
            "Div",
            inputs=[expected_path_len, cn_const_name],
            outputs=[norm_pl]
        ))

        # 5) outlier_score = 2^(- normalized_path_len)
        neg_pl = "neg_pl"
        main_nodes.append(helper.make_node(
            "Neg",
            inputs=[norm_pl],
            outputs=[neg_pl]
        ))
        two_const = "two_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[two_const],
            value=helper.make_tensor(
                name="two_tensor",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[2.0]
            )
        ))
        outlier_score = "outlier_score"
        main_nodes.append(helper.make_node(
            "Pow",
            inputs=[two_const, neg_pl],
            outputs=[outlier_score]
        ))

        # 6) Compare => predicted_label
        threshold_const = "threshold_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[threshold_const],
            value=helper.make_tensor(
                name="threshold_tensor",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(self.outlier_score_threshold)]
            )
        ))
        less_name = "less_name"
        main_nodes.append(helper.make_node(
            "Less",
            inputs=[outlier_score, threshold_const],
            outputs=[less_name]
        ))
        not_name = "not_name"
        main_nodes.append(helper.make_node(
            "Not",
            inputs=[less_name],
            outputs=[not_name]
        ))
        predicted_label = "predicted_label"
        main_nodes.append(helper.make_node(
            "Cast",
            inputs=[not_name],
            outputs=[predicted_label],
            to=TensorProto.INT32
        ))

        outlier_score_info = helper.make_tensor_value_info(
            "outlier_score", TensorProto.FLOAT, [None, 1]
        )
        predicted_label_info = helper.make_tensor_value_info(
            "predicted_label", TensorProto.INT32, [None, 1]
        )

        graph = helper.make_graph(
            name="ExtendedIF_Loop_Graph",
            inputs=[features_info],
            outputs=[outlier_score_info, predicted_label_info],
            nodes=main_nodes
        )

        model = helper.make_model(
            graph,
            producer_name="ExtendedIFLoopConverter",
            opset_imports=[helper.make_opsetid("", 13)]
        )
        return model

    def _append_tree_table_constant(self, tree_id: int, main_nodes: List[NodeProto], output_name: str):
        node_array = self.trees_data[tree_id]
        num_nodes = len(node_array)
        if num_nodes == 0:
            # Edge case: empty tree
            tensor = helper.make_tensor(
                name=output_name + "_val",
                data_type=TensorProto.FLOAT,
                dims=[1, 1],
                vals=[0.0]
            )
            cnode = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[output_name],
                value=tensor
            )
            main_nodes.append(cnode)
            return

        dim_per_node = 4 + self.num_features
        data_floats: List[float] = []
        for nd in node_array:
            leftC = float(nd['leftChild'])
            rightC = float(nd['rightChild'])
            offset = float(nd.get('offset', 0.0))
            numInst = float(nd['numInstances'])
            norm_arr = nd.get('norm', [])
            padded_norm = list(norm_arr) + [0.0]*(self.num_features - len(norm_arr))
            row = [leftC, rightC, offset, numInst] + padded_norm
            data_floats.extend(row)

        tensor = helper.make_tensor(
            name=output_name + "_val",
            data_type=TensorProto.FLOAT,
            dims=[num_nodes, dim_per_node],
            vals=data_floats
        )
        cnode = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[output_name],
            value=tensor
        )
        main_nodes.append(cnode)

    def _make_loop_node_for_tree(self, tree_id: int, node_table: str, path_len_output_name: str) -> NodeProto:
        body_graph = self._make_loop_body_graph(f"Tree_{tree_id}_loopBody", node_table)
        trip_count_const = f"tree_{tree_id}_trip_count"
        trip_count_val = 999999

        loop_node = helper.make_node(
            "Loop",
            inputs=[
                self._const_i64_scalar(trip_count_const, trip_count_val),
                "loop_cond_init",
                "loop_init_nodeId",
                "loop_init_pathLen"
            ],
            outputs=[f"ignore_nodeId_{tree_id}", path_len_output_name],
            name=f"tree_{tree_id}_loop"
        )
        loop_node.attribute.extend([
            helper.make_attribute("body", body_graph)
        ])
        return loop_node

    def _make_loop_body_graph(self, graph_name: str, node_table_name: str) -> GraphProto:
        # 4 inputs => (cond_out, curNodeId_out, pathLen_out)
        iter_in = helper.make_tensor_value_info("iter_in", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        curNodeId_in = helper.make_tensor_value_info("curNodeId_in", TensorProto.INT64, [])
        pathLen_in = helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, [])

        nodes: List[NodeProto] = []

        # (A) Unsqueeze nodeId => gather => shape [1,4+num_features]
        uq_axes = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["uq_axes"],
            value=helper.make_tensor("uq_axes_val", TensorProto.INT64, [1], [0])
        )
        nodes.append(uq_axes)

        nodeId_1d = "nodeId_1d"
        unsq_node = helper.make_node(
            "Unsqueeze",
            inputs=["curNodeId_in", "uq_axes"],
            outputs=[nodeId_1d]
        )
        nodes.append(unsq_node)

        gather_out = "nodeRow"
        gnode = helper.make_node(
            "Gather",
            inputs=[node_table_name, nodeId_1d],
            outputs=[gather_out],
            axis=0
        )
        nodes.append(gnode)

        # (B) Split => col0..3 + norms => Squeeze
        dim_count = 4 + self.num_features
        col_names = [f"col{i}" for i in range(dim_count)]
        split_node = helper.make_node(
            "Split",
            inputs=[gather_out],
            outputs=col_names,
            axis=1
        )
        nodes.append(split_node)

        leftChild_f = "leftChild_f"
        rightChild_f = "rightChild_f"
        offset_f = "offset_f"
        numInst_f = "numInst_f"
        norm_scalars = [f"norm{i}_sc" for i in range(self.num_features)]

        # We want leftChild_f, rightChild_f, offset_f, numInst_f as scalars => squeeze both dims
        # but for norms, we want shape [1], so we only squeeze axis=0 => from shape [1,1] => shape [1].
        def sqz_both(inp, out):
            cAxes = out + "_axes"
            c1 = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[cAxes],
                value=helper.make_tensor(cAxes+"_val", TensorProto.INT64, [2], [0,1])
            )
            c2 = helper.make_node(
                "Squeeze",
                inputs=[inp, cAxes],
                outputs=[out]
            )
            return [c1, c2]

        def sqz_one(inp, out):
            cAxes = out + "_axes"
            c1 = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[cAxes],
                value=helper.make_tensor(cAxes+"_val", TensorProto.INT64, [1], [0])
            )
            c2 = helper.make_node(
                "Squeeze",
                inputs=[inp, cAxes],
                outputs=[out]
            )
            return [c1, c2]

        # For the first 4:
        nodes.extend(sqz_both(col_names[0], leftChild_f))
        nodes.extend(sqz_both(col_names[1], rightChild_f))
        nodes.extend(sqz_both(col_names[2], offset_f))
        nodes.extend(sqz_both(col_names[3], numInst_f))

        for i in range(self.num_features):
            nodes.extend(sqz_one(col_names[i+4], norm_scalars[i]))

        # cast left,right => int64 => sum => eq -2 => eq_leaf
        leftChild_i = "leftChild_i"
        cast_l = helper.make_node("Cast", inputs=[leftChild_f], outputs=[leftChild_i], to=TensorProto.INT64)
        nodes.append(cast_l)

        rightChild_i = "rightChild_i"
        cast_r = helper.make_node("Cast", inputs=[rightChild_f], outputs=[rightChild_i], to=TensorProto.INT64)
        nodes.append(cast_r)

        sum_lr = "sum_lr"
        nodes.append(helper.make_node("Add", inputs=[leftChild_i, rightChild_i], outputs=[sum_lr]))

        neg2_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["neg2_const"],
            value=helper.make_tensor("neg2_val", TensorProto.INT64, [], [-2])
        )
        nodes.append(neg2_c)

        eq_leaf = "eq_leaf"
        nodes.append(helper.make_node(
            "Equal",
            inputs=[sum_lr, "neg2_const"],
            outputs=[eq_leaf]
        ))

        # If => pathLen => tmpPathLen
        if_leaf_node = helper.make_node(
            "If",
            inputs=[eq_leaf],
            outputs=["tmpPathLen"],
            name="if_leaf",
            then_branch=self._make_leaf_subgraph(),
            else_branch=self._make_notleaf_subgraph()
        )
        nodes.append(if_leaf_node)

        pathLen_next = "pathLen_next"
        nodes.append(helper.make_node(
            "Identity",
            inputs=["tmpPathLen"],
            outputs=[pathLen_next]
        ))

        # cond_out => not(eq_leaf)
        not_leaf_bool = "not_leaf_bool"
        nodes.append(helper.make_node(
            "Not",
            inputs=[eq_leaf],
            outputs=[not_leaf_bool]
        ))
        nodes.append(helper.make_node(
            "Identity",
            inputs=[not_leaf_bool],
            outputs=["cond_out"]
        ))

        # If => chooseChild => else => -1 => curNodeId_out
        if_next = helper.make_node(
            "If",
            inputs=[not_leaf_bool],
            outputs=["curNodeId_out"],
            name="if_nextNode",
            then_branch=self._make_chooseChild_subgraph(norm_scalars),
            else_branch=self._make_minusOne_subgraph()
        )
        nodes.append(if_next)

        # pathLen_out => pathLen_next
        nodes.append(helper.make_node(
            "Identity",
            inputs=[pathLen_next],
            outputs=["pathLen_out"]
        ))

        # Value infos
        extra_info = [
            helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("leftChild_f", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("rightChild_f", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("offset_f", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("leftChild_i", TensorProto.INT64, []),
            helper.make_tensor_value_info("rightChild_i", TensorProto.INT64, []),
            helper.make_tensor_value_info(eq_leaf, TensorProto.BOOL, []),
            helper.make_tensor_value_info("sum_lr", TensorProto.INT64, []),
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features]),
        ]
        for nm in norm_scalars:
            # each norm is shape [1]
            extra_info.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, [1]))

        cond_out_vi = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        curNodeId_out_vi = helper.make_tensor_value_info("curNodeId_out", TensorProto.INT64, [])
        pathLen_out_vi = helper.make_tensor_value_info("pathLen_out", TensorProto.FLOAT, [])

        body_graph = helper.make_graph(
            name=graph_name,
            nodes=nodes,
            inputs=[iter_in, cond_in, curNodeId_in, pathLen_in],
            outputs=[cond_out_vi, curNodeId_out_vi, pathLen_out_vi],
            value_info=extra_info
        )
        return body_graph

    def _make_leaf_subgraph(self) -> GraphProto:
        """
        eq_leaf => pathLen_in + avgPL(numInst_f)
        Captures pathLen_in, numInst_f from parent. No formal inputs.
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        chain = self._build_avgpathlen_inline("numInst_f", "avgpl_out")
        nodes.extend(chain)
        nodes.append(helper.make_node(
            "Add",
            inputs=["pathLen_in", "avgpl_out"],
            outputs=["subg_out"]
        ))

        subg = helper.make_graph(
            name="leafSubgraph",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            ]
        )
        return subg

    def _make_notleaf_subgraph(self) -> GraphProto:
        """
        not-leaf => pathLen_in + 1
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        one_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one_c"],
            value=helper.make_tensor("one_c_val", TensorProto.FLOAT, [], [1.0])
        )
        nodes.append(one_c)

        nodes.append(helper.make_node(
            "Add",
            inputs=["pathLen_in", "one_c"],
            outputs=["subg_out"]
        ))

        subg = helper.make_graph(
            name="notLeafSubgraph",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            ]
        )
        return subg

    def _make_chooseChild_subgraph(self, norm_names: List[str]) -> GraphProto:
        """
        not_leaf => dot = sum(norm_i * feats_squeezed) => if => left or right => cast => subg_out
        Each norm_i is shape [1]. We Concat them along axis=0 => shape [nFeatures].
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.INT64, [])
        nodes: List[NodeProto] = []

        # Squeeze features => shape [num_features]
        sqz_axes_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["chooseFeats_squeezeAxes"],
            value=helper.make_tensor("chooseFeats_squeezeAxes_val", TensorProto.INT64, [1], [0])
        )
        nodes.append(sqz_axes_c)

        feats_squeezed = "feats_squeezed"
        nodes.append(helper.make_node(
            "Squeeze",
            inputs=["features", "chooseFeats_squeezeAxes"],
            outputs=[feats_squeezed]
        ))

        # Concat the norm_i => shape [nFeatures], axis=0
        norm_concat = "norm_concat"
        nodes.append(helper.make_node(
            "Concat",
            inputs=norm_names,
            outputs=[norm_concat],
            axis=0
        ))

        mul_out = "mul_out"
        nodes.append(helper.make_node(
            "Mul",
            inputs=[norm_concat, feats_squeezed],
            outputs=[mul_out]
        ))
        dot_name = "dot_name"
        nodes.append(helper.make_node(
            "ReduceSum",
            inputs=[mul_out],
            outputs=[dot_name],
            keepdims=0
        ))

        cond_dot = "cond_dot"
        nodes.append(helper.make_node(
            "Less",
            inputs=[dot_name, "offset_f"],
            outputs=[cond_dot]
        ))

        pick_out = "pick_out"
        pick_if = helper.make_node(
            "If",
            inputs=[cond_dot],
            outputs=[pick_out],
            name="pick_if",
            then_branch=self._make_pick_subgraph(True),
            else_branch=self._make_pick_subgraph(False)
        )
        nodes.append(pick_if)

        cast_out = "cast_out"
        nodes.append(helper.make_node(
            "Cast",
            inputs=[pick_out],
            outputs=[cast_out],
            to=TensorProto.INT64
        ))
        nodes.append(helper.make_node(
            "Identity",
            inputs=[cast_out],
            outputs=["subg_out"]
        ))

        subg = helper.make_graph(
            name="chooseChildSubgraph",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                           helper.make_tensor_value_info("offset_f", TensorProto.FLOAT, []),
                           helper.make_tensor_value_info("leftChild_f", TensorProto.FLOAT, []),
                           helper.make_tensor_value_info("rightChild_f", TensorProto.FLOAT, []),
                           helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features]),
                       ] + [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in norm_names]
        )
        return subg

    def _make_pick_subgraph(self, pick_left: bool) -> GraphProto:
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.FLOAT, [])
        chosen = "leftChild_f" if pick_left else "rightChild_f"

        node = helper.make_node(
            "Identity",
            inputs=[chosen],
            outputs=["subg_out"]
        )
        subg = helper.make_graph(
            name=("pick_left" if pick_left else "pick_right"),
            nodes=[node],
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("offset_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("leftChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("rightChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features]),
            ]
        )
        return subg

    def _make_minusOne_subgraph(self) -> GraphProto:
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.INT64, [])
        node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["subg_out"],
            value=helper.make_tensor(
                name="minusOne_val",
                data_type=TensorProto.INT64,
                dims=[],
                vals=[-1]
            )
        )
        subg = helper.make_graph(
            name="elseMinusOne",
            nodes=[node],
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("offset_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("leftChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("rightChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features]),
            ]
        )
        return subg

    def _build_avgpathlen_inline(self, numInst_name: str, output_name: str) -> List[NodeProto]:
        nodes: List[NodeProto] = []

        gamma_val = 0.5772156649
        gamma_name = "euler_gamma_const"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[gamma_name],
            value=helper.make_tensor(
                name=gamma_name + "_val",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[gamma_val]
            )
        ))

        two_name = "two_val"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[two_name],
            value=helper.make_tensor(
                name=two_name + "_tens",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[2.0]
            )
        ))

        one_name = "one_val"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[one_name],
            value=helper.make_tensor(
                name=one_name + "_tens",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0]
            )
        ))

        nMinus1_name = "nMinus1"
        nodes.append(helper.make_node(
            "Sub",
            inputs=[numInst_name, one_name],
            outputs=[nMinus1_name]
        ))

        logNM1_name = "logNM1"
        nodes.append(helper.make_node(
            "Log",
            inputs=[nMinus1_name],
            outputs=[logNM1_name]
        ))

        addGamma_name = "addGamma"
        nodes.append(helper.make_node(
            "Add",
            inputs=[logNM1_name, gamma_name],
            outputs=[addGamma_name]
        ))

        partial1 = "partial1"
        nodes.append(helper.make_node(
            "Mul",
            inputs=[two_name, addGamma_name],
            outputs=[partial1]
        ))

        ratio_name = "ratio"
        nodes.append(helper.make_node(
            "Div",
            inputs=[nMinus1_name, numInst_name],
            outputs=[ratio_name]
        ))

        partial2 = "partial2"
        nodes.append(helper.make_node(
            "Mul",
            inputs=[two_name, ratio_name],
            outputs=[partial2]
        ))

        nodes.append(helper.make_node(
            "Sub",
            inputs=[partial1, partial2],
            outputs=[output_name]
        ))

        return nodes

    def _const_i64_scalar(self, name: str, val: int) -> str:
        return name + "_out"

    def _avg_path_len_formula(self, n: int) -> float:
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - (2.0 * (n - 1.0) / n)

    def convert_and_save(self, output_path: str):
        model = self.convert()
        graph = model.graph

        # Insert loop-init constants
        loop_cond_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_cond_init"],
            value=helper.make_tensor(
                name="loop_cond_init_val",
                data_type=TensorProto.BOOL,
                dims=[],
                vals=[True]
            )
        )
        loop_nodeId_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_init_nodeId"],
            value=helper.make_tensor(
                name="loop_init_nodeId_val",
                data_type=TensorProto.INT64,
                dims=[],
                vals=[0]
            )
        )
        loop_pathLen_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_init_pathLen"],
            value=helper.make_tensor(
                name="loop_init_pathLen_val",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[0.0]
            )
        )

        # define the big int for trip_count for each tree
        new_nodes = [loop_cond_node, loop_nodeId_node, loop_pathLen_node]
        for i in range(self.num_trees):
            trip_count_name = f"tree_{i}_trip_count_out"
            node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[trip_count_name],
                value=helper.make_tensor(
                    name=f"trip_count_tensor_{i}",
                    data_type=TensorProto.INT64,
                    dims=[],
                    vals=[999999]
                )
            )
            new_nodes.append(node)

        final_nodes = new_nodes + list(graph.node)
        del graph.node[:]
        graph.node.extend(final_nodes)

        onnx.checker.check_model(model)
        onnx.save_model(model, output_path)
        logger.info(f"Saved extended iForest ONNX (Loop, single-row) to {output_path}")
