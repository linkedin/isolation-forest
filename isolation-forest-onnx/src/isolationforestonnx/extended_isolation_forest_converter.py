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
    Single-row Extended Isolation Forest in ONNX.

    - Each tree => Loop => gather => check leaf => pathLen => pick child => next
    - 'curNodeId' is int64 in the main loop.
    - child IDs in the node table come as float -> we cast to int64
    - We unsqueeze final outlier_score & predicted_label => shape [1,1].
    - Leaf clamp fix: if numInstances <=1 => add 0 instead of avgPL(numInstances).
    """

    def __init__(self, model_file_path: str, metadata_file_path: str):
        from avro.datafile import DataFileReader
        from avro.io import DatumReader

        # 1) Load metadata
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
            f"node arrays in self.trees_data"
        )

    def convert(self) -> ModelProto:
        """
        Builds the ONNX graph for single-row extended isolation forest.

        Output: outlier_score shape [1,1], predicted_label shape [1,1].
        """
        features_info = helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, [None, self.num_features]
        )

        main_nodes: List[NodeProto] = []
        tree_outputs: List[str] = []

        # Build a loop for each tree => pathLen
        for i in range(self.num_trees):
            table_name = f"tree_{i}_nodes"
            self._append_tree_table_constant(i, main_nodes, table_name)

            path_name = f"tree_{i}_pathLen"
            loop_node = self._make_loop_node_for_tree(i, table_name, path_name)
            main_nodes.append(loop_node)
            tree_outputs.append(path_name)

        # Sum path lengths across all trees
        if len(tree_outputs) > 1:
            sum_pl = "sum_path_len"
            main_nodes.append(
                helper.make_node("Sum", inputs=tree_outputs, outputs=[sum_pl])
            )
        else:
            sum_pl = tree_outputs[0]

        # average => expected_path_len
        denom_name = "trees_count_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[denom_name],
            value=helper.make_tensor(
                "trees_count_val", TensorProto.FLOAT, [], [float(self.num_trees)]
            )
        ))
        expected_pl = "expected_path_len"
        main_nodes.append(helper.make_node(
            "Div",
            inputs=[sum_pl, denom_name],
            outputs=[expected_pl]
        ))

        # normalize by c(n)
        c_n_val = self._avg_path_len_formula(self.num_samples)  # e.g. c(256)
        cn_name = "cn_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[cn_name],
            value=helper.make_tensor("cn_val", TensorProto.FLOAT, [], [c_n_val])
        ))
        norm_pl = "normalized_path_len"
        main_nodes.append(helper.make_node(
            "Div",
            inputs=[expected_pl, cn_name],
            outputs=[norm_pl]
        ))

        # outlier_score_scalar = 2^(-norm_pl)
        neg_pl = "neg_pl"
        main_nodes.append(helper.make_node(
            "Neg", inputs=[norm_pl], outputs=[neg_pl]
        ))
        two_name = "two_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[two_name],
            value=helper.make_tensor(
                "two_val", TensorProto.FLOAT, [], [2.0]
            )
        ))
        outlier_score_scalar = "outlier_score_scalar"
        main_nodes.append(helper.make_node(
            "Pow",
            inputs=[two_name, neg_pl],
            outputs=[outlier_score_scalar]
        ))

        # Compare => predicted_label_scalar
        threshold_c = "threshold_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[threshold_c],
            value=helper.make_tensor(
                "threshold_val",
                TensorProto.FLOAT,
                [],
                [self.outlier_score_threshold]
            )
        ))
        less_out = "less_out"
        main_nodes.append(helper.make_node(
            "Less", inputs=[outlier_score_scalar, threshold_c], outputs=[less_out]
        ))
        not_out = "not_out"
        main_nodes.append(helper.make_node(
            "Not", inputs=[less_out], outputs=[not_out]
        ))
        predicted_label_scalar = "predicted_label_scalar"
        main_nodes.append(helper.make_node(
            "Cast",
            inputs=[not_out],
            outputs=[predicted_label_scalar],
            to=TensorProto.INT32
        ))

        # Unsqueeze => final shape [1,1] for both outlier_score & predicted_label
        out_score_axes = "out_score_axes"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[out_score_axes],
            value=helper.make_tensor(
                "out_score_axes_val",
                TensorProto.INT64,
                [2],
                [0,1]
            )
        ))
        outlier_score = "outlier_score"
        main_nodes.append(helper.make_node(
            "Unsqueeze",
            inputs=[outlier_score_scalar, out_score_axes],
            outputs=[outlier_score]
        ))

        pred_label_axes = "pred_label_axes"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[pred_label_axes],
            value=helper.make_tensor(
                "pred_label_axes_val",
                TensorProto.INT64,
                [2],
                [0,1]
            )
        ))
        predicted_label = "predicted_label"
        main_nodes.append(helper.make_node(
            "Unsqueeze",
            inputs=[predicted_label_scalar, pred_label_axes],
            outputs=[predicted_label]
        ))

        out_score_info = helper.make_tensor_value_info(
            "outlier_score", TensorProto.FLOAT, [None, 1]
        )
        pred_label_info = helper.make_tensor_value_info(
            "predicted_label", TensorProto.INT32, [None, 1]
        )

        graph = helper.make_graph(
            name="ExtendedIFSingleRowGraph",
            nodes=main_nodes,
            inputs=[features_info],
            outputs=[out_score_info, pred_label_info]
        )

        model = helper.make_model(
            graph,
            producer_name="ExtendedIFSingleRowConverter",
            opset_imports=[helper.make_opsetid("", 13)]
        )
        return model

    def _append_tree_table_constant(
            self, tree_id: int, main_nodes: List[NodeProto], output_name: str
    ):
        """
        Build a Constant node for the [num_nodes, 4 + num_features] table:
         0: leftChild (float)
         1: rightChild (float)
         2: offset (float)
         3: numInstances (float)
         4..(3+num_features): norm coords
        """
        node_array = self.trees_data[tree_id]
        n_nodes = len(node_array)
        if n_nodes == 0:
            t = helper.make_tensor(
                output_name+"_val",
                TensorProto.FLOAT,
                [1,1],
                [0.0]
            )
            c = helper.make_node(
                "Constant", inputs=[], outputs=[output_name], value=t
            )
            main_nodes.append(c)
            return

        dim_per_node = 4 + self.num_features
        vals = []
        for nd in node_array:
            leftC = float(nd['leftChild'])
            rightC = float(nd['rightChild'])
            offset = float(nd.get('offset', 0.0))
            nInst = float(nd['numInstances'])
            norm = nd.get('norm', [])
            padded = list(norm) + [0.0]*(self.num_features - len(norm))
            row = [leftC, rightC, offset, nInst] + padded
            vals.extend(row)

        t = helper.make_tensor(
            name=output_name+"_val",
            data_type=TensorProto.FLOAT,
            dims=[n_nodes, dim_per_node],
            vals=vals
        )
        c = helper.make_node(
            "Constant", inputs=[], outputs=[output_name], value=t
        )
        main_nodes.append(c)

    def _make_loop_node_for_tree(
            self, tree_id: int, table_name: str, pathLen_out: str
    ) -> NodeProto:
        sg = self._make_loop_body_graph(f"Tree_{tree_id}_loopBody", table_name)
        loop_node = helper.make_node(
            "Loop",
            inputs=[
                self._const_i64_scalar(f"tree_{tree_id}_tripCount", 999999),
                "loop_cond_init",
                "loop_init_nodeId",
                "loop_init_pathLen"
            ],
            outputs=[
                f"ignore_nodeId_{tree_id}",
                pathLen_out
            ],
            name=f"tree_{tree_id}_loop"
        )
        loop_node.attribute.extend([
            helper.make_attribute("body", sg)
        ])
        return loop_node

    def _make_loop_body_graph(
            self, graph_name: str, table_name: str
    ) -> GraphProto:
        """
        4 inputs => iter_in(int64), cond_in(bool),
                    curNodeId_in(int64), pathLen_in(float)
        3 outputs => cond_out(bool), curNodeId_out(int64), pathLen_out(float)
        """
        iter_in = helper.make_tensor_value_info("iter_in", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        nodeId_in = helper.make_tensor_value_info("curNodeId_in", TensorProto.INT64, [])
        pathLen_in = helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, [])

        nodes: List[NodeProto] = []

        # gather => shape [1, 4+num_features]
        uq_axes = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["uq_axes"],
            value=helper.make_tensor(
                "uq_axes_val",
                TensorProto.INT64,
                [1],
                [0]
            )
        )
        nodes.append(uq_axes)

        nodeId_1d = "nodeId_1d"
        un_node = helper.make_node(
            "Unsqueeze",
            inputs=["curNodeId_in", "uq_axes"],
            outputs=[nodeId_1d]
        )
        nodes.append(un_node)

        gather_out = "nodeRow"
        gnode = helper.make_node(
            "Gather",
            inputs=[table_name, nodeId_1d],
            outputs=[gather_out],
            axis=0
        )
        nodes.append(gnode)

        # Split => col0=leftChild_f, col1=rightChild_f, col2=offset_f, col3=numInst_f,...
        col_names = [f"col{i}" for i in range(4 + self.num_features)]
        split_node = helper.make_node(
            "Split",
            inputs=[gather_out],
            outputs=col_names,
            axis=1
        )
        nodes.append(split_node)

        def sqz_both(inp, out):
            cAxes = out+"_axes"
            c = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[cAxes],
                value=helper.make_tensor(
                    cAxes+"_val",
                    TensorProto.INT64,
                    [2],
                    [0,1]
                )
            )
            s = helper.make_node(
                "Squeeze",
                inputs=[inp, cAxes],
                outputs=[out]
            )
            return [c, s]

        def sqz_one(inp, out):
            cAxes = out+"_axes"
            c = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[cAxes],
                value=helper.make_tensor(
                    cAxes+"_val",
                    TensorProto.INT64,
                    [1],
                    [0]
                )
            )
            s = helper.make_node(
                "Squeeze",
                inputs=[inp, cAxes],
                outputs=[out]
            )
            return [c, s]

        leftChild_f = "leftChild_f"
        rightChild_f = "rightChild_f"
        offset_f = "offset_f"
        numInst_f = "numInst_f"
        nodes.extend(sqz_both(col_names[0], leftChild_f))
        nodes.extend(sqz_both(col_names[1], rightChild_f))
        nodes.extend(sqz_both(col_names[2], offset_f))
        nodes.extend(sqz_both(col_names[3], numInst_f))

        norm_names: List[str] = []
        for i in range(self.num_features):
            nm = f"norm{i}_sc"
            norm_names.append(nm)
            nodes.extend(sqz_one(col_names[i+4], nm))

        # cast left,right => int64 => sum => eq -2 => eq_leaf
        leftChild_i = "leftChild_i"
        nodes.append(helper.make_node(
            "Cast",
            inputs=[leftChild_f],
            outputs=[leftChild_i],
            to=TensorProto.INT64
        ))
        rightChild_i = "rightChild_i"
        nodes.append(helper.make_node(
            "Cast",
            inputs=[rightChild_f],
            outputs=[rightChild_i],
            to=TensorProto.INT64
        ))

        sum_lr = "sum_lr"
        nodes.append(helper.make_node(
            "Add",
            inputs=[leftChild_i, rightChild_i],
            outputs=[sum_lr]
        ))

        neg2_const = "neg2_const"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[neg2_const],
            value=helper.make_tensor(
                "neg2_val",
                TensorProto.INT64,
                [],
                [-2]
            )
        ))
        eq_leaf = "eq_leaf"
        nodes.append(helper.make_node(
            "Equal",
            inputs=[sum_lr, neg2_const],
            outputs=[eq_leaf]
        ))

        # If => leaf => pathLen_in + clamp(avgPL(numInst_f)) => else => pathLen_in+1 => tmpPathLen
        if_leaf = helper.make_node(
            "If",
            inputs=[eq_leaf],
            outputs=["tmpPathLen"],
            then_branch=self._make_leaf_subgraph_clamped(),
            else_branch=self._make_notleaf_subgraph(),
            name="if_leaf"
        )
        nodes.append(if_leaf)

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
        if_choose = helper.make_node(
            "If",
            inputs=[not_leaf_bool],
            outputs=["curNodeId_out"],
            name="if_chooseChild",
            then_branch=self._make_chooseChild_subgraph(
                offset_f, leftChild_f, rightChild_f, norm_names
            ),
            else_branch=self._make_minusOne_subgraph()
        )
        nodes.append(if_choose)

        nodes.append(helper.make_node(
            "Identity",
            inputs=[pathLen_next],
            outputs=["pathLen_out"]
        ))

        cond_out_vi = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        node_out_vi = helper.make_tensor_value_info("curNodeId_out", TensorProto.INT64, [])
        pl_out_vi = helper.make_tensor_value_info("pathLen_out", TensorProto.FLOAT, [])

        body_graph = helper.make_graph(
            name=graph_name,
            nodes=nodes,
            inputs=[
                helper.make_tensor_value_info("iter_in", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("curNodeId_in", TensorProto.INT64, []),
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
            ],
            outputs=[cond_out_vi, node_out_vi, pl_out_vi],
        )
        return body_graph

    def _make_leaf_subgraph_clamped(self) -> GraphProto:
        """
        eq_leaf => pathLen_out = pathLen_in + (if numInst_f>1 => avgPL(...) else => 0)
        This subgraph does a small If(numInst_f>1) => chain => else => 0
        """
        out_vi = helper.make_tensor_value_info("leaf_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        # (1) Compare numInst_f>1
        one_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one_val_leaf"],
            value=helper.make_tensor("one_val_leaf_t", TensorProto.FLOAT, [], [1.0])
        )
        nodes.append(one_c)

        gt1_name = "gt1_name"
        nodes.append(helper.make_node(
            "Greater",
            inputs=["numInst_f", "one_val_leaf"],
            outputs=[gt1_name]
        ))

        # (2) If => then => pathLen_in + avgPL(numInst_f), else => pathLen_in + 0
        sub_if_out = "sub_if_out"
        leaf_if_node = helper.make_node(
            "If",
            inputs=[gt1_name],
            outputs=[sub_if_out],
            then_branch=self._make_leafbranch_subgraph(),
            else_branch=self._make_leafelse_subgraph(),
            name="leaf_if_node"
        )
        nodes.append(leaf_if_node)

        # Identity => "leaf_out"
        nodes.append(helper.make_node(
            "Identity", inputs=[sub_if_out], outputs=["leaf_out"]
        ))

        subg = helper.make_graph(
            name="leafSubgraphClamp",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            ]
        )
        return subg

    def _make_leafbranch_subgraph(self) -> GraphProto:
        """
        THEN branch => do pathLen_in + avgPL(numInst_f)
        """
        out_vi = helper.make_tensor_value_info("leaf_branch_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        # Build the inline formula => "avgpl_out"
        chain_nodes = self._build_avgpathlen_inline("numInst_f", "avgpl_out")
        nodes.extend(chain_nodes)

        # Add => pathLen_in + avgpl_out => subg_out
        leaf_sum = "leaf_sum"
        nodes.append(helper.make_node(
            "Add", inputs=["pathLen_in", "avgpl_out"], outputs=[leaf_sum]
        ))

        # Identity => final
        nodes.append(helper.make_node(
            "Identity", inputs=[leaf_sum], outputs=["leaf_branch_out"]
        ))

        subg = helper.make_graph(
            name="leafBranch_ifGt1",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            ]
        )
        return subg

    def _make_leafelse_subgraph(self) -> GraphProto:
        """
        ELSE branch => pathLen_in + 0
        """
        out_vi = helper.make_tensor_value_info("leaf_else_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        zero_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["zero_val_leaf"],
            value=helper.make_tensor("zero_val_leaf_t", TensorProto.FLOAT, [], [0.0])
        )
        nodes.append(zero_c)

        leaf_sum2 = "leaf_sum2"
        nodes.append(helper.make_node(
            "Add", inputs=["pathLen_in", "zero_val_leaf"], outputs=[leaf_sum2]
        ))

        # Identity => final
        nodes.append(helper.make_node(
            "Identity", inputs=[leaf_sum2], outputs=["leaf_else_out"]
        ))

        subg = helper.make_graph(
            name="leafElse_ifNumInstLe1",
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
        eq_leaf=false => pathLen_in+1 => subg_out => float
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.FLOAT, [])
        nodes: List[NodeProto] = []

        one_c = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one_val"],
            value=helper.make_tensor("one_val_t", TensorProto.FLOAT, [], [1.0])
        )
        nodes.append(one_c)

        nodes.append(helper.make_node(
            "Add", inputs=["pathLen_in", "one_val"], outputs=["subg_out"]
        ))

        sg = helper.make_graph(
            name="notLeafSubgraph",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("numInst_f", TensorProto.FLOAT, []),
            ]
        )
        return sg

    def _make_chooseChild_subgraph(
            self,
            offset_f: str,
            leftChild_f: str,
            rightChild_f: str,
            norm_names: List[str]
    ) -> GraphProto:
        """
        not_leaf => dot= sum(norm_i * feats) => if dot<offset => left else => right => cast => subg_out(int64).
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.INT64, [])
        nodes: List[NodeProto] = []

        # Squeeze features => shape [num_features]
        sq_axes = "choose_feats_axes"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[sq_axes],
            value=helper.make_tensor(
                "choose_feats_axes_val",
                TensorProto.INT64,
                [1],
                [0]
            )
        ))
        feats_sq = "feats_squeezed"
        nodes.append(helper.make_node(
            "Squeeze",
            inputs=["features", sq_axes],
            outputs=[feats_sq]
        ))

        # Concat the norm => shape [num_features]
        norm_concat = "norm_concat"
        nodes.append(helper.make_node(
            "Concat",
            inputs=norm_names,
            outputs=[norm_concat],
            axis=0
        ))

        mul_out = "mul_out"
        nodes.append(helper.make_node(
            "Mul", inputs=[norm_concat, feats_sq], outputs=[mul_out]
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
            inputs=[dot_name, offset_f],
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

        # cast => int64 => subg_out
        cast_node = "cast_child"
        nodes.append(helper.make_node(
            "Cast", inputs=[pick_out], outputs=[cast_node], to=TensorProto.INT64
        ))
        nodes.append(helper.make_node(
            "Identity",
            inputs=[cast_node],
            outputs=["subg_out"]
        ))

        sg = helper.make_graph(
            name="chooseChildSubgraph",
            nodes=nodes,
            inputs=[],
            outputs=[out_vi],
            value_info=[
                           helper.make_tensor_value_info(offset_f, TensorProto.FLOAT, []),
                           helper.make_tensor_value_info(leftChild_f, TensorProto.FLOAT, []),
                           helper.make_tensor_value_info(rightChild_f, TensorProto.FLOAT, []),
                           helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features])
                       ] + [
                           helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in norm_names
                       ]
        )
        return sg

    def _make_pick_subgraph(self, pick_left: bool) -> GraphProto:
        """
        pick_left => Identity(leftChild_f) => subg_out (float)
        pick_right => Identity(rightChild_f) => subg_out (float)
        """
        out_vi = helper.make_tensor_value_info("subg_out", TensorProto.FLOAT, [])
        chosen = "leftChild_f" if pick_left else "rightChild_f"

        node = helper.make_node(
            "Identity",
            inputs=[chosen],
            outputs=["subg_out"]
        )
        sg = helper.make_graph(
            name=("pick_left" if pick_left else "pick_right"),
            nodes=[node],
            inputs=[],
            outputs=[out_vi],
            value_info=[
                helper.make_tensor_value_info("leftChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("rightChild_f", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, self.num_features]),
            ]
        )
        return sg

    def _make_minusOne_subgraph(self) -> GraphProto:
        """
        Return -1 (int64) => subg_out
        """
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
            outputs=[out_vi]
        )
        return subg

    def _build_avgpathlen_inline(
            self,
            numInst_name: str,
            output_name: str
    ) -> List[NodeProto]:
        """
        avgPL(n) = 2*(ln(n-1) + gamma) - 2*(n-1)/n
        (We only compute if n>1, but subgraph might see invalid log => NaN if not clamped.)
        """
        nodes: List[NodeProto] = []

        gamma_val = 0.5772156649
        gamma_name = "gamma_const"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[gamma_name],
            value=helper.make_tensor(
                gamma_name+"_val",
                TensorProto.FLOAT,
                [],
                [gamma_val]
            )
        ))

        two_name = "two_val"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[two_name],
            value=helper.make_tensor(
                two_name+"_val",
                TensorProto.FLOAT,
                [],
                [2.0]
            )
        ))

        one_name = "one_val"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[one_name],
            value=helper.make_tensor(
                one_name+"_val",
                TensorProto.FLOAT,
                [],
                [1.0]
            )
        ))

        nm1 = "nMinus1"
        nodes.append(helper.make_node(
            "Sub", inputs=[numInst_name, one_name], outputs=[nm1]
        ))

        log_nm1 = "log_nm1"
        nodes.append(helper.make_node(
            "Log", inputs=[nm1], outputs=[log_nm1]
        ))

        add_g = "add_g"
        nodes.append(helper.make_node(
            "Add", inputs=[log_nm1, gamma_name], outputs=[add_g]
        ))

        part1 = "part1"
        nodes.append(helper.make_node(
            "Mul", inputs=[two_name, add_g], outputs=[part1]
        ))

        ratio = "ratio"
        nodes.append(helper.make_node(
            "Div", inputs=[nm1, numInst_name], outputs=[ratio]
        ))

        part2 = "part2"
        nodes.append(helper.make_node(
            "Mul", inputs=[two_name, ratio], outputs=[part2]
        ))

        nodes.append(helper.make_node(
            "Sub", inputs=[part1, part2], outputs=[output_name]
        ))
        return nodes

    def _const_i64_scalar(self, name: str, val: int) -> str:
        """
        Return the name of the output, but not create the node here (we do that up above).
        """
        return name + "_out"

    def _avg_path_len_formula(self, n: int) -> float:
        """
        c(n) formula for the entire forest, typically n=256 or so.
        """
        if n <= 1:
            return 0.0
        return 2.0*(np.log(n - 1) + np.euler_gamma) - 2.0*((n - 1)/n)

    def convert_and_save(self, output_path: str):
        """
        Build the model, clamp leaf with numInst<=1 => +0, check, and save.
        """
        model = self.convert()
        graph = model.graph

        # Insert loop-init constants => cond_in(bool), nodeId_in(int64=0), pathLen_in(float=0)
        cond_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_cond_init"],
            value=helper.make_tensor(
                "loop_cond_init_val",
                TensorProto.BOOL,
                [],
                [True]
            )
        )
        nodeId_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_init_nodeId"],
            value=helper.make_tensor(
                "loop_init_nodeId_val",
                TensorProto.INT64,
                [],
                [0]
            )
        )
        pathLen_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["loop_init_pathLen"],
            value=helper.make_tensor(
                "loop_init_pathLen_val",
                TensorProto.FLOAT,
                [],
                [0.0]
            )
        )

        new_nodes = [cond_node, nodeId_node, pathLen_node]

        # define trip_count => 999999 for each tree
        for i in range(self.num_trees):
            tc_out = f"tree_{i}_tripCount_out"
            cnd = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[tc_out],
                value=helper.make_tensor(
                    f"trip_count_{i}_val",
                    TensorProto.INT64,
                    [],
                    [999999]
                )
            )
            new_nodes.append(cnd)

        final_nodes = new_nodes + list(graph.node)
        del graph.node[:]
        graph.node.extend(final_nodes)

        # Final check & save
        onnx.checker.check_model(model)
        onnx.save_model(model, output_path)
        logger.info(
            f"Saved extended iForest ONNX (Loop, single-row, leaf clamp) to {output_path}"
        )
