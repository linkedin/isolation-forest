package com.linkedin.relevance.isolationforest.core

import com.linkedin.relevance.isolationforest.IsolationTree
import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import com.linkedin.relevance.isolationforest.core.NodesBase.NodeBase
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

/**
 * A base trait housing code common to both standard (IsolationForest) and extended
 * isolation forest read/write logic.
 */
private[isolationforest] case object IsolationForestModelReadWriteUtils extends Logging {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   Shared NodeData constants
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  val NullNodeId: Int = -1
  val NullSplitAttribute: Int = -1
  val NullSplitValue: Double = 0.0
  val NullNumInstances: Long = -1

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

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //    Shared Metadata Helper
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /**
   * Stores the isolation forest model metadata.
   *
   * @param className The name of the saved class.
   * @param uid The immutable unique ID for the model.
   * @param timestamp The time when the model was saved.
   * @param sparkVersion The Spark version used to save the model.
   * @param params The model paramMap, as a `JValue`.
   * @param metadata All metadata, including the other fields not in paramMap.
   * @param metadataJson Full metadata file String (for debugging).
   */
  case class Metadata(
    className: String,
    uid: String,
    timestamp: Long,
    sparkVersion: String,
    params: JValue,
    metadata: JValue,
    metadataJson: String) {

    /**
     * Extract Params from metadata, and set them in the model instance. This works if all Params
     * implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
     *
     * @param instance The model instance.
     */
    def setParams(instance: Params): Unit = {

      params match {
        case JObject(pairs) =>
          pairs.foreach { case (paramName, jsonValue) =>
            val param = instance.getParam(paramName)
            val value = param.jsonDecode(compact(render(jsonValue)))
            instance.set(param, value)
          }
        case _ =>
          throw new IllegalArgumentException(s"Cannot recognize JSON metadata: ${metadataJson}.")
      }
    }
  }

  /**
   * Read a metadata JSON file from 'metadata/' subdir. The `expectedClassName` can be empty or
   * a required class name to enforce a check.
   *
   * @param path              The top-level directory path for the model
   * @param spark             The SparkSession
   * @param expectedClassName If non-empty, we verify the loaded metadata's class matches.
   */
   def loadMetadata(
    path: String,
    spark: SparkSession,
    expectedClassName: String): Metadata = {

    val metadataPath = new Path(path, "metadata").toString
    logInfo(s"Loading model metadata from $metadataPath")

    val firstLine = spark.sparkContext.textFile(metadataPath, 1).first()
    parseMetadata(firstLine, expectedClassName)
  }

  /**
   * Parse the JSON metadata string into our Metadata container.
   */
  private def parseMetadata(
    metadataStr: String,
    expectedClassName: String): Metadata = {

    implicit val fmt = DefaultFormats
    val js = parse(metadataStr)

    val cls = (js \ "class").extract[String]
    if (expectedClassName.nonEmpty) {
      require(cls == expectedClassName,
        s"Expected class $expectedClassName, but found $cls")
    }
    val uid = (js \ "uid").extract[String]
    val ts = (js \ "timestamp").extract[Long]
    val sparkVer = (js \ "sparkVersion").extract[String]
    val p = (js \ "paramMap")

    Metadata(cls, uid, ts, sparkVer, p, js, metadataStr)
  }

  /**
   * Writes the spark.ml model metadata and Params values to disk.
   *
   * @param instance The spark.ml Model instance to save.
   * @param path The path on disk used to save the metadata.
   * @param spark The SparkSession instance to use.
   * @param extraMetadata Any extra metadata to save in addition to the model Params.
   */
  def saveMetadata(
    instance: Params,
    path: String,
    spark: SparkSession,
    extraMetadata: Option[JObject] = None): Unit = {

    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, spark, extraMetadata)
    logInfo(s"Saving IsolationForestModel metadata to path ${metadataPath}")
    spark.sparkContext.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
   * This is a helper method for [[IsolationForestModelWriter.saveMetadata()]].
   *
   * @param instance The spark.ml Model instance to save.
   * @param spark The SparkSession instance to use.
   * @param extraMetadata Any extra metadata to save in addition to the model Params.
   * @return The metadata JSON string.
   */
  private def getMetadataToSave(
                                 instance: Params,
                                 spark: SparkSession,
                                 extraMetadata: Option[JObject] = None): String = {

    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq
    val jsonParams = render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList)
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> spark.sparkContext.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) => basicMetadata ~ jObject
      case None => basicMetadata
    }
    val metadataJson = compact(render(metadata))

    metadataJson
  }

  /**
   * Helper method for loading a tree ensemble from disk. This reconstructs all trees,
   * returning the root nodes.
   *
   * @param path      Path to the model directory.
   * @param spark     The SparkSession object.
   * @param buildTree A function that takes an array of NodeData and returns a NodeBase instance.
   * @return          Array of instances of type T <: IsolationTreeBase.
   * @see             `saveImpl` for how the model was saved
   */
  def loadTrees[T <: IsolationTreeBase](
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
}
