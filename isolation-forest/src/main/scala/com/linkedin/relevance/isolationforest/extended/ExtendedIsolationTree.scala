package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.IsolationTreeBase
import com.linkedin.relevance.isolationforest.core.Utils
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import com.linkedin.relevance.isolationforest.extended.ExtendedNodes.{
  ExtendedExternalNode,
  ExtendedInternalNode,
  ExtendedNode
}
import org.apache.spark.internal.Logging

import scala.annotation.tailrec
import scala.util.Random

/**
 * Represents a single Extended Isolation Tree, which uses random hyperplanes
 * (splitVector, splitOffset) to split the data at internal nodes.
 *
 * @param node The root node of this extended isolation tree.
 */
private[isolationforest] class ExtendedIsolationTree(val node: ExtendedNode)
  extends IsolationTreeBase {

  /**
   * Calculate the path length from the root node of this extended isolation tree
   * to the leaf that a particular data instance falls into.
   *
   * @param dataPoint The feature array for a single data instance.
   * @return The path length to that instance.
   */
  override def calculatePathLength(dataPoint: DataPoint): Float = {

    ExtendedIsolationTree.pathLength(dataPoint, node)
  }
}

private[isolationforest] object ExtendedIsolationTree extends Logging {

  /**
   * Fits (trains) a single extended isolation tree using random hyperplane splits.
   *
   * @param data           The array of data points used to train this tree.
   * @param randomSeed     Random seed used for reproducible hyperplane generation.
   * @param featureIndices Array of feature-column indices to use for this tree.
   * @return A trained ExtendedIsolationTree instance.
   */
  def fit(
    data: Array[DataPoint],
    randomSeed: Long,
    featureIndices: Array[Int],
    extensionLevel: Int): ExtendedIsolationTree = {

    logInfo(s"Fitting extended isolation tree with random seed $randomSeed on " +
      s"${data.length} data points and ${featureIndices.length} selected features.")

    val heightLimit = math.ceil(math.log(data.length.toDouble) / math.log(2.0)).toInt
    val rnd = new Random(randomSeed)

    val root: ExtendedNode = generateExtendedIsolationTree(
      data,
      currentHeight = 0,
      heightLimit,
      rnd,
      featureIndices,
      extensionLevel
    )

    new ExtendedIsolationTree(root)
  }

  /**
   * Recursive function that builds an extended isolation tree using random hyperplanes.
   */
  private def generateExtendedIsolationTree(
    data: Array[DataPoint],
    currentHeight: Int,
    heightLimit: Int,
    rnd: Random,
    featureIndices: Array[Int],
    extensionLevel: Int): ExtendedNode = {

    val numInstances = data.length
    if (currentHeight >= heightLimit || numInstances <= 1) {
      ExtendedExternalNode(numInstances)
    } else {
      val dim = featureIndices.length

      // extensionLevel is an Int param set by the user (0 -> standard, up to dim-1 for fully extended)
      val extensionLevelRaw = extensionLevel + 1
      
      // Then pick the final count
      val nNonZero = math.min(extensionLevelRaw, dim) // clamp so we don't exceed dimension

      // 1) Create a vector of length dim with all zeros
      val splitVector = Array.fill(dim)(0.0)

      // 2) Pick nNonZero distinct indices from [0..dim-1]
      val chosenIndices = rnd.shuffle(featureIndices.indices.toList).take(nNonZero)
      // or if you'd like to avoid double confusion with "featureIndices of featureIndices", do
      // val chosenIdxInSubspace = rnd.shuffle((0 until dim).toList).take(nNonZero)
      // then fill those positions in 'splitVector' with random Gaussians

      // fill with random gaussians
      chosenIndices.foreach { idx =>
        // idx is the index in the subspace, so if featureIndices are [3, 5, 7], idx is 0,1,2 => need to interpret carefully
        // If you want to reference the subspace index, do:
        splitVector(idx) = rnd.nextGaussian()
      }

      // Then compute dot products etc. as normal
      val dotProducts = data.map { p =>
        dot(splitVector, p, featureIndices)
      }
      val minDot = dotProducts.min
      val maxDot = dotProducts.max

      // If all points have the same dot product => leaf
      if (minDot == maxDot) {
        ExtendedExternalNode(numInstances)
      } else {
        // Pick an offset uniformly between [minDot, maxDot]
        val splitOffset = rnd.nextDouble() * (maxDot - minDot) + minDot

        // Split data into left / right
        val (leftData, rightData) = data.partition { p =>
          dot(splitVector, p, featureIndices) < splitOffset
        }

        val leftChild = generateExtendedIsolationTree(
          leftData,
          currentHeight + 1,
          heightLimit,
          rnd,
          featureIndices,
          extensionLevel
        )
        val rightChild = generateExtendedIsolationTree(
          rightData,
          currentHeight + 1,
          heightLimit,
          rnd,
          featureIndices,
          extensionLevel)

        ExtendedInternalNode(leftChild, rightChild, splitVector, splitOffset)
      }
    }
  }

  /**
   * Compute the path length for a single data point in an extended isolation tree node.
   */
  private def pathLength(dataInstance: DataPoint, node: ExtendedNode): Float = {

    @tailrec
    def recurse(currNode: ExtendedNode, depth: Float): Float = {
      currNode match {
        case ExtendedExternalNode(numInstances) =>
          // Reached leaf => add average path length factor
          depth + Utils.avgPathLength(numInstances)

        case ExtendedInternalNode(leftChild, rightChild, splitVector, splitOffset) =>
          val dotVal = dot(splitVector, dataInstance)
          if (dotVal < splitOffset) recurse(leftChild, depth + 1)
          else recurse(rightChild, depth + 1)
      }
    }

    recurse(node, 0.0f)
  }

  /**
   * Compute dot(splitVector, subsetOfFeatures).
   *
   * @param vector          The random hyperplane vector (length = featureIndices.length).
   * @param point           The data point with many features.
   * @param featureIndices  Which features from the point to use.
   * @return dot product
   */
  private def dot(
    vector: Array[Double],
    point: DataPoint,
    featureIndices: Array[Int]): Double = {

    var sum = 0.0
    var i = 0
    while (i < vector.length) {
      sum += vector(i) * point.features(featureIndices(i))
      i += 1
    }
    sum
  }

  /**
   * Compute dot(splitVector, features).
   *
   * @param vector The random hyperplane vector.
   * @param point  The data point with many features.
   * @return dot product
   */
  private def dot(
    vector: Array[Double],
    point: DataPoint): Double = {

    var sum = 0.0
    val limit = math.min(vector.length, point.features.length)
    var i = 0
    while (i < limit) {
      sum += vector(i) * point.features(i)
      i += 1
    }
    sum
  }
}
