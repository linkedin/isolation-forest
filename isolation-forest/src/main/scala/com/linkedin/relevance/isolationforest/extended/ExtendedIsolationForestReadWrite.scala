//package com.linkedin.relevance.isolationforest.extended
//
//import com.linkedin.relevance.isolationforest.core.IsolationForestModelReadWriteUtils
//import com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForestModel
//import com.linkedin.relevance.isolationforest.extended.ExtendedNodes._
//import org.apache.spark.ml.util._
//import org.apache.spark.sql.SparkSession
//import org.apache.hadoop.fs.Path
//import org.apache.spark.internal.Logging
//import org.json4s.DefaultFormats
//import org.json4s.jackson.JsonMethods._
//import org.json4s.JsonDSL._
//import org.json4s._
//
//private[extended] case object ExtendedIsolationForestModelReadWrite
//  extends IsolationForestModelReadWriteUtils
//    with Logging {
//
//  class ExtendedIsolationForestModelReader extends MLReader[ExtendedIsolationForestModel] {
//    override def load(path: String): ExtendedIsolationForestModel = {
//      implicit val fmt = DefaultFormats
//
//      val metadata = loadMetadata(path, sparkSession,
//        "com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForestModel")
//      val numSamples = (metadata.metadata \ "numSamples").extract[Int]
//      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
//      val threshold = (metadata.metadata \ "outlierScoreThreshold").extract[Double]
//
//      // Load the Avro node data
//      val dataPath = new Path(path, "data").toString
//      logInfo(s"Loading ExtendedIsolationForestModel tree data from $dataPath")
//      import sparkSession.implicits._
//
//      val allNodes = sparkSession.read.format("avro").load(dataPath)
//        .as[EnsembleNodeData]
//        .collect()
//
//      // group by treeID => build each extended tree root => wrap in ExtendedIsolationTree
//      val grouped = allNodes.groupBy(_.treeID).toSeq.sortBy(_._1)
//      val extendedTrees = grouped.map { case (_, ensembleNodes) =>
//        val nodeArray = ensembleNodes.map(_.nodeData).toArray
//        val root = buildExtendedNode(nodeArray)
//        new ExtendedIsolationTree(root)
//      }.toArray
//
//      val model = new ExtendedIsolationForestModel(metadata.uid, extendedTrees, numSamples, numFeatures)
//      metadata.setParams(model)
//      model.setOutlierScoreThreshold(threshold)
//      model
//    }
//
//    private def buildExtendedNode(data: Array[NodeData]): ExtendedNode = {
//      val sorted = data.sortBy(_.id)
//      require(sorted.map(_.id).sameElements(sorted.indices),
//        s"Extended tree load failed: node IDs not 0..N-1")
//
//      val finalNodes = new Array[ExtendedNode](sorted.length)
//      sorted.reverseIterator.foreach { nd =>
//        if (nd.leftChild == NullNodeId) {
//          // leaf => ExtendedExternalNode
//          finalNodes(nd.id) = ExtendedExternalNode(nd.numInstances)
//        } else {
//          // internal => ExtendedInternalNode
//          val left = finalNodes(nd.leftChild)
//          val right = finalNodes(nd.rightChild)
//          // In extended approach, you'd parse some random hyperplane from the node's fields if needed
//          // For now, we do a dummy placeholder
//          val dummyHyperplane = ExtendedUtils.SplitHyperplane(Array.emptyDoubleArray, 0.0)
//          finalNodes(nd.id) = ExtendedInternalNode(left, right, dummyHyperplane)
//        }
//      }
//      finalNodes.head
//    }
//  }
//
//  class ExtendedIsolationForestModelWriter(model: ExtendedIsolationForestModel) extends MLWriter {
//
//    override def saveImpl(path: String): Unit = {
//      val extra: JObject =
//        ("outlierScoreThreshold" -> model.getOutlierScoreThreshold) ~
//          ("numSamples" -> model.getNumSamples) ~
//          ("numFeatures" -> model.getNumFeatures)
//
//      saveMetadata(model, path, sparkSession, Some(extra))
//
//      val dataPath = new Path(path, "data").toString
//      logInfo(s"Saving ExtendedIsolationForestModel tree data to $dataPath")
//
//      val rdd = sparkSession.sparkContext
//        .parallelize(model.extendedIsolationTrees.zipWithIndex)
//        .flatMap { case (exTree, idx) =>
//          // Convert to NodeData using the same ensemble logic from the base
//          EnsembleNodeData.build(exTree.extendedNode, idx)
//        }
//
//      import sparkSession.implicits._
//      rdd.toDF().repartition(1).write.format("avro").save(dataPath)
//    }
//  }
//}
