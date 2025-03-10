package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.IsolationTreeBase
import com.linkedin.relevance.isolationforest.core.Utils
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import com.linkedin.relevance.isolationforest.extended.ExtendedNodes.{
  ExtendedExternalNode,
  ExtendedInternalNode,
  ExtendedNode
}
import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import org.apache.spark.internal.Logging

import scala.annotation.tailrec
import scala.util.Random

/**
 * A single Extended Isolation Tree, which uses random hyperplanes
 * defined by (splitVector, splitOffset).
 *
 * @param node The root node of this extended isolation tree.
 */
private[isolationforest] class ExtendedIsolationTree(val node: ExtendedNode)
  extends IsolationTreeBase {

  /**
   * Calculate the path length from the root node of this extended isolation tree
   * to the leaf that a particular data instance falls into.
   */
  override def calculatePathLength(dataPoint: DataPoint): Float = {
    ExtendedIsolationTree.pathLength(dataPoint, node)
  }
}

private[isolationforest] object ExtendedIsolationTree extends Logging {

  /**
   * Trains a single extended isolation tree using random hyperplane splits:
   *  - We choose extensionLevel+1 non-zero coordinates (clamped to the subspace dimension),
   *  - fill them with random Gaussian values,
   *  - compute dot-products of all data points,
   *  - pick an offset in [minDot, maxDot].
   */
  def fit(
           data: Array[DataPoint],
           randomSeed: Long,
           featureIndices: Array[Int],
           extensionLevel: Int
         ): ExtendedIsolationTree = {

    logInfo(s"Fitting extended isolation tree (random seed=$randomSeed) on " +
      s"${data.length} points, ${featureIndices.length} subspace-dim, extensionLevel=$extensionLevel")

    val heightLimit = math.ceil(math.log(data.length.toDouble) / math.log(2.0)).toInt
    val rnd = new Random(randomSeed)

    val root = generateExtendedIsolationTree(
      data, currentHeight = 0, heightLimit, rnd, featureIndices, extensionLevel
    )
    new ExtendedIsolationTree(root)
  }

  /**
   * Recursively build the extended isolation tree.
   */
  private def generateExtendedIsolationTree(
    data: Array[DataPoint],
    currentHeight: Int,
    heightLimit: Int,
    rnd: Random,
    featureIndices: Array[Int],
    extensionLevel: Int): ExtendedNode = {

    val numInstances = data.length
    val numFeatures = data.head.features.length

    if (currentHeight >= heightLimit || numInstances <= 1) {
      // Leaf node
      ExtendedExternalNode(numInstances)
    } else {
      val dim = featureIndices.length
      // We allow up to (extensionLevel+1) coordinates to be non-zero
      val nNonZero = math.min(extensionLevel + 1, dim)

      // Build the hyperplane norm vector for the full space
      val splitVector = Array.fill(numFeatures)(0.0)

      // Randomly choose which feature indices become non-zero
      val chosenFeatureIndices = rnd.shuffle((0 until dim).toList).take(nNonZero)
      chosenFeatureIndices.foreach { i =>
        splitVector(featureIndices(i)) = rnd.nextGaussian()
      }

      // Compute the L2 norm of the vector
      val squaredSum = splitVector.map(x => x * x).sum
      val normValue = math.sqrt(squaredSum)

      // Check to ensure the norm is non-zero to avoid division by zero
      if (normValue == 0) {
        throw new IllegalArgumentException("Cannot normalize a zero vector.")
      }

      // Normalize the vector
      val normSplitVector = splitVector.map(_ / normValue)

//      println("\n\n==========================================" +
//        "\nextensionLevel: " + extensionLevel +
//        "\nnumFeatures: " + numFeatures +
//        "\nfeatureIndices: " + featureIndices.mkString(", ") +
//        "\ndim: " + dim +
//        "\nnNonZero: " + nNonZero +
//        "\nchosenFeatureIndices: " + chosenFeatureIndices.mkString(", ") +
//        "\nsplitVector: " + splitVector.mkString(", ") +
//        "\nnormSplitVector: " + normSplitVector.mkString(", ") +
//       "\n===================================================\n\n")

      // Compute dot products in this subspace
      val dotProducts = data.map(p => dot(normSplitVector, p))
      val minDot = dotProducts.min
      val maxDot = dotProducts.max

      // If all dot-values are the same => leaf
      if (minDot == maxDot) {
        ExtendedExternalNode(numInstances)
      } else {
        // pick offset uniformly in [minDot, maxDot]
        val splitOffset = rnd.nextDouble() * (maxDot - minDot) + minDot

        // Partition into left (dot < offset) and right (>= offset)
        val (leftData, rightData) = data.partition { point =>
          dot(normSplitVector, point) < splitOffset
        }
        // If one side is empty, we can just treat as leaf
        if (leftData.isEmpty || rightData.isEmpty) {
          ExtendedExternalNode(numInstances)
        } else {
          // Build children
          val leftChild = generateExtendedIsolationTree(
            leftData, currentHeight + 1, heightLimit, rnd, featureIndices, extensionLevel
          )
          val rightChild = generateExtendedIsolationTree(
            rightData, currentHeight + 1, heightLimit, rnd, featureIndices, extensionLevel
          )
          ExtendedInternalNode(leftChild, rightChild, SplitHyperplane(normSplitVector, splitOffset))
        }
      }
    }
  }

  /**
   * Compute path length for a single point, recursing until we hit a leaf.
   */
  private def pathLength(dataPoint: DataPoint, node: ExtendedNode): Float = {

    @tailrec
    def recurse(curr: ExtendedNode, depth: Float): Float = {
      curr match {
        case ExtendedExternalNode(numInstances) =>
          depth + Utils.avgPathLength(numInstances) // Leaf

        case ExtendedInternalNode(left, right, splitHyperplane) =>
          val dpVal = dot(splitHyperplane.norm, dataPoint)
          if (dpVal < splitHyperplane.offset) recurse(left, depth + 1)
          else recurse(right, depth + 1)
      }
    }

    recurse(node, 0.0f)
  }

  /**
   * Compute the dot product between the subspace `splitVector` and the data point's selected features.
   *
   * @param splitVector  array of length == featureIndices.length
   * @param point        data point (float[] features)
   * @return dot product
   */
  private def dot(splitVector: Array[Double], point: DataPoint): Double = {
    var sum = 0.0
    var i = 0
    while (i < splitVector.length) {
      sum += splitVector(i) * point.features(i)
      i += 1
    }
    sum
  }
}
