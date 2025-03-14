package com.linkedin.relevance.isolationforest.core

import com.linkedin.relevance.isolationforest.core.IsolationForestModelReadWriteUtils.{
  EnsembleNodeData,
  NodeData,
  NullNodeId,
  loadMetadata,
  loadTrees,
  saveMetadata
}
import com.linkedin.relevance.isolationforest.{IsolationForestModel, IsolationTree}
import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.util._
import org.apache.spark.sql.SparkSession
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods._
import org.json4s.JsonDSL._
import org.json4s._

private[isolationforest] case object IsolationForestModelReadWrite extends Logging {

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
   * Reader for the standard IsolationForestModel
   */
  class IsolationForestModelReader extends MLReader[IsolationForestModel] {

    override def load(path: String): IsolationForestModel = {
      implicit val format = DefaultFormats
      val expectedClassName = "com.linkedin.relevance.isolationforest.IsolationForestModel"

      val metadata = loadMetadata(path, sparkSession, expectedClassName)
      val numSamples = (metadata.metadata \ "numSamples").extract[Int]
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val threshold = (metadata.metadata \ "outlierScoreThreshold").extract[Double]

      val rootNodes = loadTrees[IsolationTree](path, sparkSession, buildTreeFromNodes)
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

  /**
   * Writer for the standard IsolationForestModel
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
          ("numSamples", instance.getNumSamples) ~
          ("numFeatures", instance.getNumFeatures)
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

      saveMetadata(instance, path, spark, Some(extraMetadata))
      val dataPath = new Path(path, "data").toString
      val nodeDataRDD = spark.sparkContext.parallelize(instance.isolationTrees.zipWithIndex.toIndexedSeq)
        .flatMap { case (tree, treeID) => EnsembleNodeData.build(tree, treeID) }
      logInfo(s"Saving IsolationForestModel tree data to path ${dataPath}")
      spark.createDataFrame(nodeDataRDD)
        .repartition(1)
        .write
        .format("avro")
        .save(dataPath)
    }
  }
}
