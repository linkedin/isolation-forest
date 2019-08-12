package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import com.linkedin.relevance.isolationforest.Utils.DataPoint
import org.apache.spark.internal.Logging

import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer
import scala.util.Random


/**
  * A trained isolation tree.
  *
  * @param node The root node of the isolation tree model.
  */
private[isolationforest] class IsolationTree(val node: Node) extends Serializable {

  import IsolationTree._

  /**
    * Returns the path length from the root node of this isolation tree to the node in the tree that
    * contains a particular data point.
    *
    * @param dataInstance The feature array for a single data instance.
    * @return The path length to the instance.
    */
  private[isolationforest] def calculatePathLength(dataInstance: DataPoint): Float =
    pathLength(dataInstance, node)
}

/**
  * Companion object used to train the IsolationTree class.
  */
private[isolationforest] case object IsolationTree extends Logging {

  /**
    * Fits a single isolation tree to the input data.
    *
    * @param data The 2D array containing the feature values (columns) for the data instances (rows)
    *             used to train this particular isolation tree.
    * @param randomSeed The random seed used to generate this tree.
    * @param featureIndices Array containing the feature column indices used for training this
    *                       particular tree.
    * @return A trained isolation tree object.
    */
  def fit(data: Array[DataPoint], randomSeed: Long, featureIndices: Array[Int]): IsolationTree = {

    logInfo(s"Fitting isolation tree with random seed ${randomSeed} on" +
    s" ${featureIndices.seq.toString} features (indices) using ${data.length} data points.")

    def log2(x: Double): Double = math.log10(x) / math.log10(2.0)
    val heightLimit = math.ceil(log2(data.length.toDouble)).toInt

    new IsolationTree(
      generateIsolationTree(
        data,
        heightLimit,
        new Random(randomSeed),
        featureIndices))
  }

  /**
   * Generates an isolation tree. It encloses the generateIsolationTreeInternal() method to hide the
   * currentTreeHeight parameter.
   *
   * @param data Feature data used to generate the isolation tree.
   * @param heightLimit The tree height at which the algorithm terminates.
   * @param randomState The random state object.
   * @param featureIndices Array containing the feature column indices used for training this
   *                       particular tree.
   * @return The root node of the isolation tree.
   */
  def generateIsolationTree(
    data: Array[DataPoint],
    heightLimit: Int,
    randomState: Random,
    featureIndices: Array[Int]): Node = {

    /**
     * This is a recursive method that generates an isolation tree. It is an implementation of the
     * iTree(X,e,l) algorithm in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
     *
     * @param data Feature data used to generate the isolation tree.
     * @param currentTreeHeight Height of the current tree. Initialize this to 0 for a new tree.
     * @param heightLimit The tree height at which the algorithm terminates.
     * @param randomState The random state object.
     * @param featureIndices Array containing the feature column indices used for training this
     *                       particular tree.
     * @return The root node of the isolation tree.
     */
    def generateIsolationTreeInternal(
      data: Array[DataPoint],
      currentTreeHeight: Int,
      heightLimit: Int,
      randomState: Random,
      featureIndices: Array[Int]): Node = {

      /**
        * Randomly selects a feature and feature value to split upon.
        *
        * @param data The data at the particular node in question.
        * @return Tuple containing the feature index and the split value. Feature index is -1 if no
        *         features could be split.
        */
      def getFeatureToSplit(data: Array[DataPoint]): (Int, Double) = {

        val availableFeatures = featureIndices.to[ListBuffer]
        var foundFeature = false
        var featureIndex = -1
        var featureSplitValue = 0.0

        // Randomly select a feature to split on and determine the split value between the current
        // minimum and maximum values of that feature among current data points in that node. If the
        // minimum and maximum values are the same then move on to the next randomly selected
        // feature.
        while (!foundFeature && availableFeatures.nonEmpty) {
          val featureIndexTrial = availableFeatures
            .remove(randomState.nextInt(availableFeatures.length))
          val featureValues = data.map(x => x.features(featureIndexTrial))
          val minFeatureValue = featureValues.min.toDouble
          val maxFeatureValue = featureValues.max.toDouble

          if (minFeatureValue != maxFeatureValue) {
            foundFeature = true
            featureIndex = featureIndexTrial
            featureSplitValue = ((maxFeatureValue - minFeatureValue) * randomState.nextDouble
              + minFeatureValue)
          }
        }
        (featureIndex, featureSplitValue)
      }

      val (featureIndex, featureSplitValue) = getFeatureToSplit(data)
      val numInstances = data.length

      if (featureIndex == -1 || currentTreeHeight >= heightLimit || numInstances <= 1)
        ExternalNode(numInstances)
      else {
        val dataLeft = data.filter(x => x.features(featureIndex) < featureSplitValue)
        val dataRight = data.filter(x => x.features(featureIndex) >= featureSplitValue)

        InternalNode(
          generateIsolationTreeInternal(dataLeft, currentTreeHeight + 1, heightLimit, randomState, featureIndices),
          generateIsolationTreeInternal(dataRight, currentTreeHeight + 1, heightLimit, randomState, featureIndices),
          featureIndex,
          featureSplitValue)
      }
    }

    generateIsolationTreeInternal(data, 0, heightLimit, randomState, featureIndices)
  }

  /**
    * Returns the path length from the root node of an isolation tree to the node in the tree that
    * contains a particular data point.
    *
    * @param dataInstance      A single data point for scoring.
    * @param node              The root node of the tree used to calculate the path length.
    * @return The path length to the instance.
    */
  def pathLength(dataInstance: DataPoint, node: Node): Float = {

    /**
      * This recursive method returns the path length from a node of an isolation tree to the node
      * in the tree that contains a particular data point. The returned path length includes an
      * additional component dependent upon how many training data points ended up in this node. This
      * is the PathLength(x,T,e) algorithm in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
      *
      * @param dataInstance      A single data point for scoring.
      * @param node              The root node of the tree used to calculate the path length.
      * @param currentPathLength The path length to the current node.
      * @return The path length to the instance.
      */
    @tailrec
    def pathLengthInternal(dataInstance: DataPoint, node: Node, currentPathLength: Float): Float = {

      node match {
        case externalNode: ExternalNode =>
          currentPathLength + Utils.avgPathLength(externalNode.numInstances)
        case internalNode: InternalNode =>
          val splitAttribute = internalNode.splitAttribute
          val splitValue = internalNode.splitValue
          if (dataInstance.features(splitAttribute) < splitValue) {
            pathLengthInternal(dataInstance, internalNode.leftChild, currentPathLength + 1)
          } else {
            pathLengthInternal(dataInstance, internalNode.rightChild, currentPathLength + 1)
          }
      }
    }

    pathLengthInternal(dataInstance, node, 0)
  }
}
