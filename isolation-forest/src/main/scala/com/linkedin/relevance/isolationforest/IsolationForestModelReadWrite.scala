/*
 * This file uses modified code from the spark.ml files treeModels.scala,
 * RandomForestClassifier.scala, and ReadWrite.scala, which are open sourced under the Apache 2.0
 * license.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.json4s.{DefaultFormats, JObject, JValue}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}


/**
  * Contains the IsolationForestModelReader and IsolationForestModelWriter classes and supporting
  * case classes. A trained IsolationForestModel can be written (read) to (from) HDFS using these
  * classes.
  */
private[isolationforest] case object IsolationForestModelReadWrite extends Logging {

  val NullNodeId: Int = -1
  val NullSplitAttribute: Int = -1
  val NullSplitValue: Double = 0.0
  val NullNumInstances: Long = -1

  /**
    * Reads a saved isolation forest model from disk.
    */
  class IsolationForestModelReader extends MLReader[IsolationForestModel] with Serializable {

    /**
      * Overrides [[org.apache.spark.ml.util.MLReader.load]] in order to load an
      * [[IsolationForestModel]] instance.
      *
      * @param path The path to the saved isolation forest model.
      * @return The loaded IsolationForestModel instance.
      */
    override def load(path: String): IsolationForestModel = {

      implicit val format = DefaultFormats
      val (metadata, treesData) = loadImpl(path, sparkSession)
      val numSamples = (metadata.metadata \ "numSamples").extract[Int]
      val outlierScoreThreshold = (metadata.metadata \ "outlierScoreThreshold").extract[Double]

      val trees = treesData.map {
        case internalNode: InternalNode => new IsolationTree(internalNode.asInstanceOf[InternalNode])
        case externalNode: ExternalNode => new IsolationTree(externalNode.asInstanceOf[ExternalNode])
      }

      val model = new IsolationForestModel(metadata.uid, trees, numSamples)
      metadata.setParams(model)
      model.setOutlierScoreThreshold(outlierScoreThreshold)

      model
    }

    /**
      * Helper method for loading an isolation tree ensemble from disk. This reconstructs all trees,
      * returning the root nodes.
      *
      * @param path  Path given to `saveImpl`
      * @param spark The SparkSession object.
      *
      * @return (ensemble metadata, array of root nodes each tree), where the root node is linked
      *         with all descendents
      * @see `saveImpl` for how the model was saved
      */
    private def loadImpl(path: String, spark: SparkSession): (Metadata, Array[Node]) = {

      import spark.implicits._

      val metadata = loadMetadata(
        path,
        spark,
        "com.linkedin.relevance.isolationforest.IsolationForestModel")

      val dataPath = new Path(path, "data").toString
      logInfo(s"Loading IsolationForestModel tree data from path ${dataPath}")
      val nodeData = spark.read
        .format("com.databricks.spark.avro")
        .load(dataPath)
        .as[EnsembleNodeData]
      val rootNodesRDD = nodeData.rdd
        .map(ensembleNodeData => (ensembleNodeData.treeID, ensembleNodeData.nodeData))
        .groupByKey()
        .map { case (treeID, node) => treeID -> buildTreeFromNodes(node.toArray) }
      val rootNodes = rootNodesRDD.sortByKey().values.collect()

      (metadata, rootNodes)
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
      * Load metadata saved using [[IsolationForestModelWriter.saveMetadata()]].
      *
      * @param path The path to the saved metadata.
      * @param spark The SparkSession instance.
      * @param expectedClassName If non empty, this is checked against the loaded metadata.
      */
    private def loadMetadata(
      path: String,
      spark: SparkSession,
      expectedClassName: String = ""): Metadata = {

      val metadataPath = new Path(path, "metadata").toString
      logInfo(s"Loading IsolationForestModel metadata from path ${metadataPath}")
      val metadataStr = spark.sparkContext.textFile(metadataPath, 1).first()
      parseMetadata(metadataStr, expectedClassName)
    }

    /**
      * Parse metadata JSON string produced by [[IsolationForestModelWriter.getMetadataToSave()]].
      * This is a helper for [[IsolationForestModelReader.loadMetadata()]].
      *
      * @param metadataStr JSON string of metadata.
      * @param expectedClassName If non empty, this is checked against the loaded metadata.
      * @return A [[Metadata]] instance built from the parsed metadata string.
      */
    private def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {

      val metadata = parse(metadataStr)

      implicit val format = DefaultFormats
      // The "\" used below is an operator that allows us to query json4s.JValue fields by name.
      val className = (metadata \ "class").extract[String]
      val uid = (metadata \ "uid").extract[String]
      val timestamp = (metadata \ "timestamp").extract[Long]
      val sparkVersion = (metadata \ "sparkVersion").extract[String]
      val params = metadata \ "paramMap"
      if (expectedClassName.nonEmpty) {
        require(className == expectedClassName, s"Error loading metadata: Expected class name" +
          s" $expectedClassName but found class name $className")
      }

      Metadata(className, uid, timestamp, sparkVersion, params, metadata, metadataStr)
    }
  }

  /**
    * Writes a saved isolation forest model to disk.
    *
    * @param instance The IsolationForestModel instance to write to disk.
    */
  class IsolationForestModelWriter(instance: IsolationForestModel) extends MLWriter {

    /**
      * Overrides [[org.apache.spark.ml.util.MLWriter.saveImpl]].
      *
      * @param path The file path to the directory where the saved model should be written.
      */
    override def saveImpl(path: String): Unit = {

      val extraMetadata: JObject =
        ("outlierScoreThreshold", instance.getOutlierScoreThreshold) ~
          ("numSamples", instance.getNumSamples)
      saveImplHelper(path, sparkSession, extraMetadata)
    }

    /**
      * Helper method for saving a tree ensemble to disk.
      *
      * @param path The path on disk used to save the ensemble model.
      * @param spark The SparkSession instance to use.
      * @param extraMetadata Metadata such as outlierScoreThreshold and numSamples.
      */
    private def saveImplHelper(path: String, spark: SparkSession, extraMetadata: JObject): Unit = {

      saveMetadata(instance, path, spark, Some(extraMetadata))
      val dataPath = new Path(path, "data").toString
      val nodeDataRDD = spark.sparkContext.parallelize(instance.isolationTrees.zipWithIndex)
        .flatMap { case (tree, treeID) => EnsembleNodeData.build(tree, treeID) }
      logInfo(s"Saving IsolationForestModel tree data to path ${dataPath}")
      spark.createDataFrame(nodeDataRDD)
        .repartition(1)
        .write
        .format("com.databricks.spark.avro")
        .save(dataPath)
    }

    /**
      * Writes the spark.ml model metadata and Params values to disk.
      *
      * @param instance The spark.ml Model instance to save.
      * @param path The path on disk used to save the metadata.
      * @param spark The SparkSession instance to use.
      * @param extraMetadata Any extra metadata to save in addition to the model Params.
      */
    private def saveMetadata(
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
      val params = instance.extractParamMap.toSeq
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
  }

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
    * Stores the data for an ensemble of trees constructed of [[Node]]s.
    *
    * @param treeID The ID specifying the tree to which this node belongs.
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
}
