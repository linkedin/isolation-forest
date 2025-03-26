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
    Demonstration of using ONNX "Loop" to implement multi-feature hyperplane splits
    in an Extended Isolation Forest.

    Key Points:
      - We have two state variables in the loop: (curNodeId, pathLen).
        The subgraph body has 3 outputs: [cond_out, curNodeId_out, pathLen_out].
      - The top-level loop node must produce 2 final outputs for these states.
      - Each tree references a separate "tree_i_trip_count_out" so multi-tree
        topological ordering works properly.
      - Squeeze must pass axes as a second input in opset13, not an attribute.
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
        """
        Builds a partial ONNX model referencing loop inputs. We do NOT call checker.check_model here.
        The final check is done in convert_and_save().
        """
        features_info = helper.make_tensor_value_info(
            "features", TensorProto.FLOAT, [None, self.num_features]
        )

        main_nodes: List[NodeProto] = []
        tree_outputs = []

        # 1) Build a Loop node for each tree
        for i in range(self.num_trees):
            # First define the node table constant
            node_table_const_name = f"tree_{i}_nodes"
            self._append_tree_table_constant(i, main_nodes, node_table_const_name)

            # The loop will produce two final states => [finalCurNodeId, finalPathLen].
            dummy_cur_node = f"tree_{i}_dummyCurNode"
            path_len_output_name = f"tree_{i}_pathLen"

            loop_node = self._make_loop_node_for_tree(
                tree_id=i,
                node_table=node_table_const_name,
                output_names=[dummy_cur_node, path_len_output_name]
            )
            main_nodes.append(loop_node)

            # We only need pathLen for summation
            tree_outputs.append(path_len_output_name)

        # 2) If multiple trees => sum => sum_path_len
        if len(tree_outputs) > 1:
            sum_path_name = "sum_path_len"
            main_nodes.append(helper.make_node(
                "Sum",
                inputs=tree_outputs,
                outputs=[sum_path_name]
            ))
        else:
            sum_path_name = tree_outputs[0]

        # 3) average path len => expected_path_len
        expected_path_len = "expected_path_len"
        denom_const_name = "trees_count_const"
        main_nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[denom_const_name],
            value=helper.make_tensor(
                name=denom_const_name+"_val",
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
                name=cn_const_name+"_val",
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

        # Final outputs
        outlier_score_info = helper.make_tensor_value_info(
            'outlier_score', TensorProto.FLOAT, [None, 1]
        )
        predicted_label_info = helper.make_tensor_value_info(
            'predicted_label', TensorProto.INT32, [None, 1]
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
            opset_imports=[helper.make_opsetid('', 13)]
        )
        logger.info(
            "Built partial ONNX model with 2-state loop. No checker call yet."
        )
        return model

    def convert_and_save(self, output_path: str):
        """
        Build partial model => define loop constants => final check => save
        """
        model = self.convert()
        graph = model.graph

        # We must prepend loop init nodes and trip_count nodes
        existing_nodes = list(graph.node)
        graph.ClearField("node")

        # 1) Shared loop init states
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

        # 2) One trip_count constant per tree
        trip_count_nodes = []
        for i in range(self.num_trees):
            node_name = f"tree_{i}_trip_count_out"
            big_const = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[node_name],
                value=helper.make_tensor(
                    name=f"trip_count_tensor_{i}",
                    data_type=TensorProto.INT64,
                    dims=[],
                    vals=[999999]
                )
            )
            trip_count_nodes.append(big_const)

        new_nodes = [loop_cond_node, loop_nodeId_node, loop_pathLen_node] + trip_count_nodes

        # Prepend them
        graph.node.extend(new_nodes)
        graph.node.extend(existing_nodes)

        # Now do the final check
        onnx.checker.check_model(model)
        onnx.save_model(model, output_path)
        logger.info(f"Saved extended iForest ONNX to {output_path}")

    def _append_tree_table_constant(
            self,
            tree_id: int,
            main_nodes: List[NodeProto],
            output_name: str
    ):
        """
        Creates a 2D constant node with shape=[num_nodes, 4 + num_features], storing
         col0=leftChild, col1=rightChild, col2=offset, col3=numInstances, col4..=norm
        """
        node_array = self.trees_data[tree_id]
        num_nodes = len(node_array)
        if num_nodes == 0:
            # trivial
            tensor = helper.make_tensor(
                name=output_name+"_val",
                data_type=TensorProto.FLOAT,
                dims=[1,1],
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
            # pad if needed
            padded_norm = list(norm_arr) + [0.0]*(self.num_features - len(norm_arr))

            row = [leftC, rightC, offset, numInst] + padded_norm
            data_floats.extend(row)

        tensor = helper.make_tensor(
            name=output_name+"_val",
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

    def _make_loop_node_for_tree(
            self,
            tree_id: int,
            node_table: str,
            output_names: List[str]
    ) -> NodeProto:
        """
        Each loop has 4 inputs: [trip_count, cond_init, nodeId_init, pathLen_init]
        and 2 final outputs: [finalCurNodeId, finalPathLen].
        """
        body_graph = self._make_loop_body_graph(f"Tree_{tree_id}_loopBody", node_table)

        loop_node = helper.make_node(
            "Loop",
            inputs=[
                self._const_i64_scalar(f"tree_{tree_id}_trip_count", 999999),
                "loop_cond_init",
                "loop_init_nodeId",
                "loop_init_pathLen"
            ],
            outputs=output_names,
            name=f"tree_{tree_id}_loop"
        )
        loop_node.attribute.extend([
            helper.make_attribute("body", body_graph)
        ])
        return loop_node

    def _make_loop_body_graph(self, graph_name: str, node_table_name: str) -> GraphProto:
        """
        Loop body subgraph with 4 inputs => 3 outputs:
          inputs: iter_in(int64), cond_in(bool), curNodeId_in(int64), pathLen_in(float)
          outputs: cond_out(bool), curNodeId_out(int64), pathLen_out(float)
        """
        iter_in = helper.make_tensor_value_info("iter_in", TensorProto.INT64, [])
        cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
        curNodeId_in = helper.make_tensor_value_info("curNodeId_in", TensorProto.INT64, [])
        pathLen_in = helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, [])

        cond_out_info = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
        curNodeId_out_info = helper.make_tensor_value_info("curNodeId_out", TensorProto.INT64, [])
        pathLen_out_info = helper.make_tensor_value_info("pathLen_out", TensorProto.FLOAT, [])

        nodes: List[NodeProto] = []

        # axes for Squeeze
        axes_name = "axes_0_1"
        axes_tensor = helper.make_tensor(
            name="axes_tensor_0_1",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[0,1]
        )
        axes_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[axes_name],
            value=axes_tensor
        )
        nodes.append(axes_const_node)

        # gather => shape [1, 4+num_features]
        gather_node_out = "nodeRow"
        nodes.append(helper.make_node(
            "Gather",
            inputs=[node_table_name, "curNodeId_in"],
            outputs=[gather_node_out],
            axis=0
        ))

        # split => columns
        split_cols = [f"col{i}" for i in range(4 + self.num_features)]
        nodes.append(helper.make_node(
            "Split",
            inputs=[gather_node_out],
            outputs=split_cols,
            axis=1
        ))

        leftChild_f = "leftChild_f"
        rightChild_f = "rightChild_f"
        offset_name = "offset"
        numInst_name = "numInstances"
        norm_outs = [f"norm{i}" for i in range(self.num_features)]

        # squeeze each column => pass axes
        nodes.append(self._squeeze_2d(split_cols[0], leftChild_f, axes_name))
        nodes.append(self._squeeze_2d(split_cols[1], rightChild_f, axes_name))
        nodes.append(self._squeeze_2d(split_cols[2], offset_name, axes_name))
        nodes.append(self._squeeze_2d(split_cols[3], numInst_name, axes_name))
        for i in range(self.num_features):
            nodes.append(self._squeeze_2d(split_cols[i+4], norm_outs[i], axes_name))

        # cast leftChild/rightChild => int64 => check leaf
        leftChild_i = "leftChild_i"
        rightChild_i = "rightChild_i"
        nodes.append(helper.make_node(
            "Cast",
            inputs=[leftChild_f],
            outputs=[leftChild_i],
            to=TensorProto.INT64
        ))
        nodes.append(helper.make_node(
            "Cast",
            inputs=[rightChild_f],
            outputs=[rightChild_i],
            to=TensorProto.INT64
        ))

        sum_lr = "sumLR"
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
                name="neg2_val",
                data_type=TensorProto.INT64,
                dims=[],
                vals=[-2]
            )
        ))
        eq_leaf = "eq_leaf"
        nodes.append(helper.make_node(
            "Equal",
            inputs=[sum_lr, neg2_const],
            outputs=[eq_leaf]
        ))

        # If(leaf) => pathLen+= avgPL(numInst), else => +1
        leaf_sg = self._make_leaf_subgraph("leafSubgraph")
        not_leaf_sg = self._make_notleaf_subgraph("notLeafSubgraph")
        if_leaf_node = helper.make_node(
            "If",
            inputs=[eq_leaf],
            outputs=["tmpPathLen"],
            then_branch=leaf_sg,
            else_branch=not_leaf_sg
        )
        nodes.append(if_leaf_node)

        pathLen_next = "pathLen_next"
        nodes.append(helper.make_node(
            "Identity",
            inputs=["tmpPathLen"],
            outputs=[pathLen_next]
        ))

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

        # nextNode => If(not_leaf) => dot<offset => left or right
        nextNode_if_sg = self._build_nextNode_subgraph(
            "chooseChildSubgraph", offset_name, norm_outs, leftChild_f, rightChild_f
        )
        if_next_node = helper.make_node(
            "If",
            inputs=[not_leaf_bool],
            outputs=["curNodeId_out"],
            then_branch=nextNode_if_sg,
            else_branch=self._make_const_int_subgraph("elseNodeMinusOne", -1)
        )
        nodes.append(if_next_node)

        # pathLen_out => pathLen_next
        nodes.append(helper.make_node(
            "Identity",
            inputs=[pathLen_next],
            outputs=["pathLen_out"]
        ))

        body_graph = helper.make_graph(
            name=graph_name,
            nodes=nodes,
            inputs=[iter_in, cond_in, curNodeId_in, pathLen_in],
            outputs=[cond_out_info, curNodeId_out_info, pathLen_out_info]
        )
        return body_graph

    def _squeeze_2d(self, inp: str, out: str, axes_input: str) -> NodeProto:
        """
        In opset13, we pass axes as a second input to Squeeze
        """
        return helper.make_node(
            "Squeeze",
            inputs=[inp, axes_input],
            outputs=[out]
        )

    def _build_nextNode_subgraph(self,
                                 name: str,
                                 offset_name: str,
                                 norm_outs: List[str],
                                 leftC: str,
                                 rightC: str) -> GraphProto:
        """
        dot= sum(norm[i]*features[i]), if dot<offset => leftC else rightC => cast => nextNode
        """
        offset_in = helper.make_tensor_value_info(offset_name, TensorProto.FLOAT, [])
        left_in = helper.make_tensor_value_info(leftC, TensorProto.FLOAT, [])
        right_in = helper.make_tensor_value_info(rightC, TensorProto.FLOAT, [])
        feats_in = helper.make_tensor_value_info("subg_features", TensorProto.FLOAT, [None, self.num_features])
        nextNode_out = helper.make_tensor_value_info("nextNode", TensorProto.INT64, [])

        nodes: List[NodeProto] = []

        # define a local axes= [0,1]
        axes_name = "axes_0_1_subg"
        axes_tensor = helper.make_tensor(
            name="axes_tensor_0_1_subg",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[0,1]
        )
        axes_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[axes_name],
            value=axes_tensor
        )
        nodes.append(axes_const_node)

        # Concat norms => shape [num_features]
        norm_concat_out = "norm_concat"
        nodes.append(helper.make_node(
            "Concat",
            inputs=norm_outs,
            outputs=[norm_concat_out],
            axis=0
        ))

        # Squeeze features => shape [num_features]
        feats_squeezed = "feats_squeezed"
        nodes.append(self._squeeze_2d("subg_features", feats_squeezed, axes_name))

        # dot => Mul => ReduceSum
        mul_out = "mul_out"
        nodes.append(helper.make_node(
            "Mul",
            inputs=[norm_concat_out, feats_squeezed],
            outputs=[mul_out]
        ))
        dot_name = "dot_name"
        nodes.append(helper.make_node(
            "ReduceSum",
            inputs=[mul_out],
            outputs=[dot_name],
            keepdims=0
        ))

        # cond => dot < offset
        cond_dot = "cond_dot"
        nodes.append(helper.make_node(
            "Less",
            inputs=[dot_name, offset_in.name],
            outputs=[cond_dot]
        ))

        # If => pick left or right => cast => nextNode
        if_node_out = "tmp_nextNodeF"
        pick_left = self._make_identity_subgraph("pick_left_subg", leftC)
        pick_right = self._make_identity_subgraph("pick_right_subg", rightC)
        if_node = helper.make_node(
            "If",
            inputs=[cond_dot],
            outputs=[if_node_out],
            then_branch=pick_left,
            else_branch=pick_right
        )
        nodes.append(if_node)

        cast_out = "cast_nextNode"
        nodes.append(helper.make_node(
            "Cast",
            inputs=[if_node_out],
            outputs=[cast_out],
            to=TensorProto.INT64
        ))
        nodes.append(helper.make_node(
            "Identity",
            inputs=[cast_out],
            outputs=["nextNode"]
        ))

        subg = helper.make_graph(
            name=name,
            inputs=[offset_in, left_in, right_in, feats_in]
                   + [helper.make_tensor_value_info(x, TensorProto.FLOAT, []) for x in norm_outs],
            outputs=[nextNode_out],
            nodes=nodes
        )
        return subg

    def _make_leaf_subgraph(self, name: str) -> GraphProto:
        pathLen_in = helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, [])
        numInst_in = helper.make_tensor_value_info("numInstances", TensorProto.FLOAT, [])
        pathLen_out = helper.make_tensor_value_info("pathLen_out", TensorProto.FLOAT, [])

        nodes: List[NodeProto] = []

        # inline formula => avgpl
        avgpl_out = "avgpl_out"
        chain_nodes = self._build_avgpathlen_inline("numInstances", avgpl_out)
        nodes.extend(chain_nodes)

        add_node = helper.make_node(
            "Add",
            inputs=["pathLen_in", avgpl_out],
            outputs=["pathLen_out"]
        )
        nodes.append(add_node)

        subg = helper.make_graph(
            name=name,
            inputs=[pathLen_in, numInst_in],
            outputs=[pathLen_out],
            nodes=nodes
        )
        return subg

    def _make_notleaf_subgraph(self, name: str) -> GraphProto:
        pathLen_in = helper.make_tensor_value_info("pathLen_in", TensorProto.FLOAT, [])
        numInst_in = helper.make_tensor_value_info("numInstances", TensorProto.FLOAT, [])
        pathLen_out = helper.make_tensor_value_info("pathLen_out", TensorProto.FLOAT, [])

        one_val = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one_val"],
            value=helper.make_tensor(
                name="one_val_tensor",
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0]
            )
        )
        add_node = helper.make_node(
            "Add",
            inputs=["pathLen_in", "one_val"],
            outputs=["pathLen_out"]
        )

        subg = helper.make_graph(
            name=name,
            inputs=[pathLen_in, numInst_in],
            outputs=[pathLen_out],
            nodes=[one_val, add_node]
        )
        return subg

    def _build_avgpathlen_inline(self, numInst_name: str, output_name: str) -> List[NodeProto]:
        """
        avgPL(n) = 2*(ln(n-1)+gamma) - 2*(n-1)/n ignoring n<=1
        """
        nodes: List[NodeProto] = []

        gamma_val = 0.5772156649
        gamma_name = "euler_gamma_const"
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[gamma_name],
            value=helper.make_tensor(
                name=gamma_name+"_val",
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
                name=two_name+"_tens",
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
                name=one_name+"_tens",
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

    def _make_identity_subgraph(self, name: str, input_name: str) -> GraphProto:
        vin = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [])
        vout = helper.make_tensor_value_info("out_val", TensorProto.FLOAT, [])
        node = helper.make_node("Identity", inputs=[input_name], outputs=["out_val"])
        return helper.make_graph([node], name, [vin], [vout])

    def _make_const_int_subgraph(self, name: str, val: int) -> GraphProto:
        out_info = helper.make_tensor_value_info("out_val", TensorProto.INT64, [])
        node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["out_val"],
            value=helper.make_tensor(
                name=name+"_val",
                data_type=TensorProto.INT64,
                dims=[],
                vals=[val]
            )
        )
        return helper.make_graph([node], name, [], [out_info])

    def _const_i64_scalar(self, name: str, val: int) -> str:
        """
        Return the symbolic name, e.g. "tree_0_trip_count_out".
        We'll define one for each tree in convert_and_save().
        """
        return name+"_out"

    def _avg_path_len_formula(self, n: int) -> float:
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n