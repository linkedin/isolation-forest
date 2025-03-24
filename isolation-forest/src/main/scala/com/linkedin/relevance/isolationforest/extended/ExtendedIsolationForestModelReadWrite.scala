package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.IsolationForestModel
import com.linkedin.relevance.isolationforest.core.IsolationForestModelReadWriteUtils.{NullNodeId, NullNumInstances, loadMetadata, saveMetadata}
import com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForestModel
import com.linkedin.relevance.isolationforest.extended.ExtendedNodes._
import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import org.apache.spark.ml.util._
import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._


private[extended] case object ExtendedIsolationForestModelReadWrite extends Logging {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   ExtendedNodeData constants
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  val NullNorm: Array[Double] = Array.emptyDoubleArray
  val NullOffset: Double = 0.0

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Data classes for serializing extended trees to Avro
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /**
   * Data record for a single node in an extended isolation tree.
   *
   * @param id           Node index (pre-order traversal). Must be unique in [0..N-1].
   * @param leftChild    Index of the left child, or -1 if leaf.
   * @param rightChild   Index of the right child, or -1 if leaf.
   * @param norm         Random hyperplane normal vector. Empty if leaf.
   * @param offset       Random hyperplane offset. 0.0 if leaf.
   * @param numInstances For a leaf node, the number of samples that landed here. Otherwise -1.
   */
  case class ExtendedNodeData(
                               id: Int,
                               leftChild: Int,
                               rightChild: Int,
                               norm: Array[Double],
                               offset: Double,
                               numInstances: Long
                             )

  case object ExtendedNodeData {

    /**
     * Serializes a binary extended isolation tree.
     *
     * @param node The head node (ExtendedNode) of the tree to be serialized.
     * @return Serialized sequence of ExtendedNodeData instances in pre-order traversal.
     */
    def build(node: ExtendedNode): Seq[ExtendedNodeData] = {

      /**
       * This helper method recursively traverses an extended isolation tree in pre-order,
       * assigning node IDs from 0..N-1.
       *
       * @param node The current ExtendedNode being visited.
       * @param id   The node index (pre-order).
       * @return A tuple of (serialized nodes in this subtree, next available ID after finishing this subtree).
       */
      def buildInternal(node: ExtendedNode, id: Int): (Seq[ExtendedNodeData], Int) = {

        node match {
          case ExtendedInternalNode(leftChild, rightChild, splitHyperplane) =>
            // Recursively build the left subtree
            val (leftNodeData, leftIdx) = buildInternal(leftChild, id + 1)
            // Recursively build the right subtree
            val (rightNodeData, rightIdx) = buildInternal(rightChild, leftIdx + 1)

            // For internal nodes, numInstances is -1 because a hyperplane-split node doesn't store that count
            val thisNodeData = ExtendedNodeData(
              id = id,
              leftChild = leftNodeData.head.id,
              rightChild = rightNodeData.head.id,
              norm = splitHyperplane.norm,
              offset = splitHyperplane.offset,
              numInstances = NullNumInstances
            )
            // Prepend this node's data, then append children
            (thisNodeData +: (leftNodeData ++ rightNodeData), rightIdx)

          case ExtendedExternalNode(numInst) =>
            // Leaf node: leftChild/rightChild = -1, norm is empty, offset=0, numInstances is stored
            val leafData = ExtendedNodeData(
              id = id,
              leftChild = NullNodeId,
              rightChild = NullNodeId,
              norm = NullNorm,
              offset = NullOffset,
              numInstances = numInst
            )
            (Seq(leafData), id)

          case _ =>
            throw new IllegalArgumentException(s"Unknown node type: ${node.getClass.toString}")
        }
      }

      // Kick off recursion from the root with ID=0
      val (serialized, _) = buildInternal(node, 0)
      serialized
    }
  }

  /**
   * Associates node data with a particular tree in the ensemble.
   *
   * @param treeID           ID (0-based) of the tree this node belongs to.
   * @param extendedNodeData The node data itself.
   */
  case class ExtendedEnsembleNodeData(
                                       treeID: Int,
                                       extendedNodeData: ExtendedNodeData
                                     )

  /**
   * Companion object to the ExtendedEnsembleNodeData class.
   */
  case object ExtendedEnsembleNodeData {

    /**
     * Serializes an [[ExtendedIsolationTree]] instance.
     *
     * @param tree   The ExtendedIsolationTree to serialize.
     * @param treeID The ID specifying the index of this isolation tree in the ensemble.
     * @return A sequence of ExtendedEnsembleNodeData instances.
     */
    def build(tree: ExtendedIsolationTree, treeID: Int): Seq[ExtendedEnsembleNodeData] = {
      // Use ExtendedNodeData.build(...) to serialize the tree's root node
      val nodeDataSeq = ExtendedNodeData.build(tree.extendedNode)
      // Map each ExtendedNodeData to ExtendedEnsembleNodeData, tagging with treeID
      nodeDataSeq.map(nd => ExtendedEnsembleNodeData(treeID, nd))
    }
  }

  /**
   * Builds a binary extended isolation tree from an array of ExtendedNodeData. The node IDs must
   * have been assigned via a pre-order traversal (0..N-1).
   *
   * @param data An Array of ExtendedNodeData describing one extended isolation tree.
   * @return The root ExtendedNode of the reconstructed tree.
   */
  private def buildExtendedNode(data: Array[ExtendedNodeData]): ExtendedNode = {
    // 1) Sort the input by ID and ensure IDs are exactly [0..length-1].
    val sorted = data.sortBy(_.id)
    require(sorted.map(_.id).sameElements(sorted.indices),
      "Extended tree load failed: node IDs must be 0..N-1 in ascending order.")

    // 2) Create a finalNodes array to hold each rebuilt node by index.
    val finalNodes = new Array[ExtendedNode](sorted.length)

    // 3) Build from the back so children are guaranteed to be ready before we construct the parent.
    sorted.reverseIterator.foreach { nd =>
      if (nd.leftChild == NullNodeId && nd.rightChild == NullNodeId) {
        // Leaf node => ExtendedExternalNode with numInstances
        finalNodes(nd.id) = ExtendedExternalNode(nd.numInstances)
      } else {
        // Internal node => ExtendedInternalNode with a random hyperplane
        val left = finalNodes(nd.leftChild)
        val right = finalNodes(nd.rightChild)
        val hyperplane = SplitHyperplane(nd.norm, nd.offset)
        finalNodes(nd.id) = ExtendedInternalNode(left, right, hyperplane)
      }
    }

    // 4) The node with id=0 is the root of the reconstructed tree
    finalNodes.head
  }

  /**
   * Writer for the ExtendedIsolationForestModel, mirroring the style of the standard
   * IsolationForestModelWriter.
   *
   * @param model The ExtendedIsolationForestModel instance to write.
   */
  class ExtendedIsolationForestModelWriter(model: ExtendedIsolationForestModel)
    extends MLWriter with Logging {

    /**
     * Main entry point for saving the model. Spark ML calls this automatically.
     *
     * @param path The file path to the directory where the model should be written.
     */
    override def saveImpl(path: String): Unit = {
      // 1) Build up any extra metadata we want to save
      val extraMetadata: JObject =
        ("outlierScoreThreshold" -> model.getOutlierScoreThreshold) ~
          ("numSamples" -> model.getNumSamples) ~
          ("numFeatures" -> model.getNumFeatures)

      // 2) Delegate to a helper function that does the actual writing
      saveImplHelper(path, sparkSession, extraMetadata)
    }

    /**
     * Helper method for saving the extended isolation forest ensemble to disk,
     * closely mirroring the approach in the standard IsolationForestModelWriter.
     *
     * @param path          The path on disk used to save the model.
     * @param spark         The current SparkSession.
     * @param extraMetadata Additional metadata to store, e.g. threshold, numSamples, etc.
     */
    private def saveImplHelper(path: String, spark: SparkSession, extraMetadata: JObject): Unit = {

      saveMetadata(model, path, spark, Some(extraMetadata))
      val dataPath = new Path(path, "data").toString
      val nodeDataRDD = spark.sparkContext.parallelize(model.extendedIsolationTrees.zipWithIndex.toIndexedSeq)
        .flatMap { case (tree, treeID) => ExtendedEnsembleNodeData.build(tree, treeID) }
      logInfo(s"Saving ExtendedIsolationForestModel tree data to $dataPath")

      import spark.implicits._
      nodeDataRDD.toDF().repartition(1).write.format("avro").save(dataPath)
    }
  }

  /**
   * Reader for the standard IsolationForestModel
   */
  class ExtendedIsolationForestModelReader extends MLReader[ExtendedIsolationForestModel] {

    /**
     * Helper method for loading an extended isolation forest ensemble from disk.
     * It reads the Avro node data, groups by treeID, applies the provided buildTree
     * function to reconstruct a root ExtendedNode for each tree, and returns them.
     *
     * @param path      Path to the model directory.
     * @param spark     The SparkSession object.
     * @param buildTree A function that takes an array of ExtendedNodeData and returns an ExtendedNode.
     * @return An Array of ExtendedNode, each the root node of one extended isolation tree.
     */
    private def loadTrees(
                           path: String,
                           spark: SparkSession,
                           buildTree: Array[ExtendedNodeData] => ExtendedNode
                         ): Array[ExtendedNode] = {

      import spark.implicits._

      val dataPath = new Path(path, "data").toString
      logInfo(s"Loading extended tree data from path $dataPath")

      // Read Avro data into Dataset[ExtendedEnsembleNodeData]
      val ds = spark.read
        .format("avro")
        .load(dataPath)
        .as[ExtendedEnsembleNodeData]

      // Group each tree's nodes by treeID, then call buildTree(...) to produce a root node
      val rootNodesRDD = ds.rdd
        .map(e => (e.treeID, e.extendedNodeData))
        .groupByKey()
        .map { case (treeID, nodeDataIter) =>
          val nodeDataArr = nodeDataIter.toArray
          treeID -> buildTree(nodeDataArr)
        }

      // Sort by treeID to keep them in ascending order and collect
      val rootNodes = rootNodesRDD.sortByKey().values.collect()
      rootNodes
    }

    /**
     * Load an ExtendedIsolationForestModel from a given path.
     *
     * @param path The directory path containing metadata/ and data/ subdirs.
     * @return A fully reconstructed ExtendedIsolationForestModel.
     */
    override def load(path: String): ExtendedIsolationForestModel = {
      implicit val format = DefaultFormats

      // 1) Load metadata and verify the class name
      val expectedClassName = "com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForestModel"
      val metadata = loadMetadata(path, sparkSession, expectedClassName)

      // 2) Extract basic parameters from metadata
      val numSamples = (metadata.metadata \ "numSamples").extract[Int]
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val threshold = (metadata.metadata \ "outlierScoreThreshold").extract[Double]
      //    val extensionLevel = (metadata.metadata \ "extensionLevel").extractOpt[Int]

      // 3) Load & rebuild each extended tree, returning an array of ExtendedNode
      val rootNodes = loadTrees(path, sparkSession, buildExtendedNode)

      // Wrap each root ExtendedNode into an ExtendedIsolationTree
      val extendedTrees = rootNodes.map(root => new ExtendedIsolationTree(root))

      // 4) Create the ExtendedIsolationForestModel
      val model = new ExtendedIsolationForestModel(
        uid = metadata.uid,
        extendedIsolationTrees = extendedTrees,
        numSamples = numSamples,
        numFeatures = numFeatures
      )

      // 5) Restore spark.ml Params from metadata, then set the outlier threshold
      metadata.setParams(model)
      model.setOutlierScoreThreshold(threshold)

      model
    }
  }
}