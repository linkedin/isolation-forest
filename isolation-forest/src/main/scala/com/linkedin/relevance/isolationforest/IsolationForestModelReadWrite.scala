package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.IsolationForestModelReadWriteUtils.{
  NullNodeId,
  NullNumInstances,
  loadMetadata,
  saveMetadata
}
import com.linkedin.relevance.isolationforest.{IsolationForestModel, IsolationTree}
import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import com.linkedin.relevance.isolationforest.core.IsolationTreeBase
import com.linkedin.relevance.isolationforest.core.NodesBase.NodeBase
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.util._
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._
import org.json4s._


private[isolationforest] case object IsolationForestModelReadWrite extends Logging {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   NodeData constants
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  val NullSplitAttribute: Int = -1
  val NullSplitValue: Double = 0.0

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   NodeData + EnsembleNodeData
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /**
   * Stores the serializable data for a [[Node]].
   *
   * @param id Node index used for tree reconstruction. Indices follow a pre-order traversal.
   * @param leftChild Left child node index, or value of -1 if this node is a leaf node.
   * @param rightChild Right child node index, or value of -1 if this node is a leaf node.
   * @param splitAttribute The feature upon which the split was performed, or -1 if this node is a
   *                       leaf node.
   * @param splitValue The feature value used for the split, or 0.0 if this node is a leaf node.
   * @param numInstances If this is a leaf node, the number of instances in for this node, or -1 if
   *                     this is an internal node.
   */
  case class NodeData(
                       id: Int,
                       leftChild: Int,
                       rightChild: Int,
                       splitAttribute: Int,
                       splitValue: Double,
                       numInstances: Long)

  /**
   * Companion object to the NodeData class.
   */
  case object NodeData {

    /**
     * Serializes a binary tree of [[Node]] instances.
     *
     * @param node The head node of the binary tree to be serialized.
     * @return Serialized sequence of NodeData instances
     */
    def build(node: Node): Seq[NodeData] = {

      /**
       * This helper method for [[NodeData.build()]] serializes a binary tree of [[Node]]
       * instances.
       *
       * @param node The head node of the binary tree to be serialized.
       * @param id Node index used for tree reconstruction. Indices follow a pre-order traversal.
       * @return (sequence of nodes in pre-order traversal order, largest ID in subtree)
       */
      def buildInternal(node: Node, id: Int): (Seq[NodeData], Int) = {

        node match {
          case internalNode: InternalNode =>
            val (leftNodeData, leftIdx) = buildInternal(internalNode.leftChild, id + 1)
            val (rightNodeData, rightIdx) = buildInternal(internalNode.rightChild, leftIdx + 1)
            // For internal nodes, numInstances doesn't exist, so it is set to -1 for serialization.
            val thisNodeData = NodeData(
              id,
              leftNodeData.head.id,
              rightNodeData.head.id,
              internalNode.splitAttribute,
              internalNode.splitValue,
              NullNumInstances)
            (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)
          case externalNode: ExternalNode =>
            // For external nodes, leftChild, rightChild, splitAttribute, and splitValue do not
            // exist, so they are set to -1, -1, -1, and 0.0, respectively, for serialization.
            (Seq(NodeData(
              id,
              NullNodeId,
              NullNodeId,
              NullSplitAttribute,
              NullSplitValue,
              externalNode.numInstances)), id)
          case _ =>
            throw new IllegalArgumentException(s"Unknown node type: ${node.getClass.toString}")
        }
      }

      val serializedNodeData = buildInternal(node, 0)
      serializedNodeData._1
    }
  }

  /**
   * Associates a NodeData with a treeID so we can store multiple trees in a single dataset.
   */
  /**
   * Stores the data for an ensemble of trees constructed of [[Node]]s.
   *
   * @param treeID   The ID specifying the tree to which this node belongs.
   * @param nodeData The [[NodeData]] instance containing the information from the corresponding
   *                 [[Node]] in the tree.
   */
  case class EnsembleNodeData(treeID: Int, nodeData: NodeData)

  /**
   * Companion object to the EnsembleNodeData class.
   */
  case object EnsembleNodeData {

    /**
     * Serializes an [[IsolationTree]] instance.
     *
     * @param tree The IsolationTree instance to serialize.
     * @param treeID The ID specifying the index of this isolation tree in the ensemble.
     * @return A sequence of EnsembleNodeData instances.
     */
    def build(tree: IsolationTree, treeID: Int): Seq[EnsembleNodeData] = {
      val nodeData = NodeData.build(tree.node)
      nodeData.map(nodeData => EnsembleNodeData(treeID, nodeData))
    }
  }

  /**
   * Builds a binary tree given an array of NodeData instances. The node IDs must
   * have been assigned via pre-order traversal.
   *
   * @param data An Array of NodeData instances.
   * @return The root node of the resulting binary tree.
   */
  private def buildTreeFromNodes(data: Array[NodeData]): Node = {

    val nodes = data.sortBy(_.id)

    require(nodes.map(x => x.id).sameElements(nodes.indices), s"Isolation tree load failed." +
      s" Expected the ${nodes.length} node IDs to be monotonically increasing from 0 to" +
      s" ${nodes.length - 1}.")

    // We fill `finalNodes` in reverse order. Since node IDs are assigned via a pre-order
    // traversal, this guarantees that child nodes will be built before parent nodes.
    val finalNodes = new Array[Node](nodes.length)
    nodes.reverseIterator.foreach { nodeData: NodeData =>
      val node = if (nodeData.leftChild != NullNodeId) {
        val leftChild = finalNodes(nodeData.leftChild)
        val rightChild = finalNodes(nodeData.rightChild)
        InternalNode(
          leftChild,
          rightChild,
          nodeData.splitAttribute,
          nodeData.splitValue)
      } else {
        ExternalNode(nodeData.numInstances)
      }
      finalNodes(nodeData.id) = node
    }

    finalNodes.head
  }

  /**
   * Writer for the standard IsolationForestModel
   */
  class IsolationForestModelWriter(model: IsolationForestModel) extends MLWriter {

    /**
     * Overrides [[org.apache.spark.ml.util.MLWriter.saveImpl]].
     *
     * @param path The file path to the directory where the saved model should be written.
     */
    override def saveImpl(path: String): Unit = {

      val extraMetadata: JObject =
        ("outlierScoreThreshold", model.getOutlierScoreThreshold) ~
          ("numSamples", model.getNumSamples) ~
          ("numFeatures", model.getNumFeatures)
      saveImplHelper(path, sparkSession, extraMetadata)
    }

    /**
     * Helper method for saving a tree ensemble to disk.
     *
     * @param path The path on disk used to save the ensemble model.
     * @param spark The SparkSession instance to use.
     * @param extraMetadata Metadata such as outlierScoreThreshold, numSamples, and numFeatures.
     */
    private def saveImplHelper(path: String, spark: SparkSession, extraMetadata: JObject): Unit = {

      saveMetadata(model, path, spark, Some(extraMetadata))
      val dataPath = new Path(path, "data").toString
      val nodeDataRDD = spark.sparkContext.parallelize(model.isolationTrees.zipWithIndex.toIndexedSeq)
        .flatMap { case (tree, treeID) => EnsembleNodeData.build(tree, treeID) }
      logInfo(s"Saving IsolationForestModel tree data to path ${dataPath}")

      import spark.implicits._
      nodeDataRDD.toDF().repartition(1).write.format("avro").save(dataPath)
    }
  }

  /**
   * Reader for the standard IsolationForestModel
   */
  class IsolationForestModelReader extends MLReader[IsolationForestModel] {

    /**
     * Helper method for loading a tree ensemble from disk. This reconstructs all trees,
     * returning the root nodes.
     *
     * @param path      Path to the model directory.
     * @param spark     The SparkSession object.
     * @param buildTree A function that takes an array of NodeData and returns a NodeBase instance.
     */
    private def loadTrees(
                           path: String,
                           spark: SparkSession,
                           buildTree: Array[NodeData] => NodeBase): Array[NodeBase] = {

      import spark.implicits._

      val dataPath = new Path(path, "data").toString
      logInfo(s"Loading tree data from path ${dataPath}")
      val nodeData = spark.read
        .format("avro")
        .load(dataPath)
        .as[EnsembleNodeData]
      val rootNodesRDD = nodeData.rdd
        .map(ensembleNodeData => (ensembleNodeData.treeID, ensembleNodeData.nodeData))
        .groupByKey()
        .map { case (treeID, node) => treeID -> buildTree(node.toArray) }
      val rootNodes = rootNodesRDD.sortByKey().values.collect()

      rootNodes
    }

    override def load(path: String): IsolationForestModel = {
      implicit val format = DefaultFormats
      val expectedClassName = "com.linkedin.relevance.isolationforest.IsolationForestModel"

      val metadata = loadMetadata(path, sparkSession, expectedClassName)
      val numSamples = (metadata.metadata \ "numSamples").extract[Int]
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val threshold = (metadata.metadata \ "outlierScoreThreshold").extract[Double]

      val rootNodes = loadTrees(path, sparkSession, buildTreeFromNodes)
      val trees = rootNodes.map {
        case internalNode: InternalNode => new IsolationTree(internalNode.asInstanceOf[InternalNode])
        case externalNode: ExternalNode => new IsolationTree(externalNode.asInstanceOf[ExternalNode])
      }

      val model = new IsolationForestModel(metadata.uid, trees, numSamples, numFeatures)
      metadata.setParams(model)
      model.setOutlierScoreThreshold(threshold)

      model
    }
  }
}
