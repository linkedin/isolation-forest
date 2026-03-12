package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.IsolationTreeBase
import com.linkedin.relevance.isolationforest.core.Utils
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import com.linkedin.relevance.isolationforest.extended.ExtendedNodes.{
  ExtendedExternalNode,
  ExtendedInternalNode,
  ExtendedNode,
}
import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector

import scala.annotation.tailrec
import scala.util.Random

/**
 * A trained extended isolation tree.
 *
 * @param extendedNode
 *   The root node of this extended isolation tree.
 */
private[isolationforest] class ExtendedIsolationTree(val extendedNode: ExtendedNode)
    extends IsolationTreeBase {

  /**
   * Returns the path length from the root node of this isolation tree to the node in the tree that
   * contains a particular data point.
   *
   * @param dataInstance
   *   The feature array for a single data instance.
   * @return
   *   The path length to the instance.
   */
  override def calculatePathLength(dataInstance: DataPoint): Float =
    ExtendedIsolationTree.pathLength(dataInstance, extendedNode)

  /**
   * Returns the path length from the root node of this isolation tree to the node in the tree that
   * contains a particular Spark vector.
   *
   * @param features
   *   The feature vector for a single data instance.
   * @return
   *   The path length to the instance.
   */
  def calculatePathLength(features: Vector): Float =
    ExtendedIsolationTree.pathLength(features, extendedNode)
}

private[isolationforest] object ExtendedIsolationTree extends Logging {

  /**
   * Trains a single extended isolation tree using random hyperplane splits.
   *
   * @param data
   *   The 2D array containing the feature values (columns) for the data instances (rows) used to
   *   train this particular isolation tree.
   * @param randomSeed
   *   The random seed used to generate this tree.
   * @param featureIndices
   *   Array containing the feature column indices used for training this particular tree.
   * @return
   *   A trained isolation tree object.
   */
  def fit(
    data: Array[DataPoint],
    randomSeed: Long,
    featureIndices: Array[Int],
    extensionLevel: Int,
  ): ExtendedIsolationTree = {

    logInfo(
      s"Fitting extended isolation tree (random seed=$randomSeed) on " +
        s"${data.length} points, ${featureIndices.length} subspace-dim, extensionLevel=$extensionLevel",
    )

    def log2(x: Double): Double = math.log10(x) / math.log10(2.0)
    val heightLimit = math.ceil(log2(data.length.toDouble)).toInt

    val rnd = new Random(randomSeed)

    val root = generateExtendedIsolationTree(
      data,
      heightLimit,
      rnd,
      featureIndices,
      extensionLevel,
    )
    new ExtendedIsolationTree(root)
  }

  /**
   * Generates an extended isolation tree. It encloses the generateExtendedIsolationTreeInternal()
   * method to hide the currentTreeHeight parameter.
   *
   * @param data
   *   Feature data used to generate the isolation tree.
   * @param heightLimit
   *   The tree height at which the algorithm terminates.
   * @param randomState
   *   The random state object.
   * @param featureIndices
   *   Array containing the feature column indices used for training this particular tree.
   * @param extensionLevel
   *   "Extension level for the random hyperplane. extensionLevel+1 = number of non-zero
   *   coordinates. 0 => axis-aligned EIF splits, dimensionOfSubspace-1 => fully extended splits"
   * @return
   *   The root node of the isolation tree.
   */
  def generateExtendedIsolationTree(
    data: Array[DataPoint],
    heightLimit: Int,
    randomState: Random,
    featureIndices: Array[Int],
    extensionLevel: Int,
  ): ExtendedNode = {

    /**
     * This is a recursive method that generates an extended isolation tree.
     *
     * @param data
     *   Feature data used to generate the isolation tree.
     * @param currentTreeHeight
     *   Height of the current tree. Initialize this to 0 for a new tree.
     * @param heightLimit
     *   The tree height at which the algorithm terminates.
     * @param randomState
     *   The random state object.
     * @param featureIndices
     *   Array containing the feature column indices used for training this particular tree.
     * @param extensionLevel
     *   "Extension level for the random hyperplane. extensionLevel+1 = number of non-zero
     *   coordinates. 0 => axis-aligned EIF splits, dimensionOfSubspace-1 => fully extended splits"
     * @return
     *   The root node of the isolation tree.
     */
    def generateExtendedIsolationTreeInternal(
      data: Array[DataPoint],
      currentTreeHeight: Int,
      heightLimit: Int,
      randomState: Random,
      featureIndices: Array[Int],
      extensionLevel: Int,
    ): ExtendedNode = {

      val numInstances = data.length

      // Base case: matches EIF paper Algorithm 2 — |X| ≤ 1 OR height limit reached.
      // numInstances=0 is possible when a degenerate split sends all points to one side.
      if (currentTreeHeight >= heightLimit || numInstances <= 1) {
        ExtendedExternalNode(numInstances)
      } else {
        val dim = featureIndices.length
        // We allow up to (extensionLevel+1) coordinates to be non-zero
        val nNonZero = math.min(extensionLevel + 1, dim)

        // Randomly choose which feature indices become non-zero (in the selected subspace)
        val chosenFeatureIndices = randomState.shuffle((0 until dim).toList).take(nNonZero)

        val sparseIndices = new Array[Int](nNonZero)
        val rawWeights = new Array[Double](nNonZero)

        var i = 0
        while (i < chosenFeatureIndices.length) {
          val featureIndexInSubspace = chosenFeatureIndices(i)
          sparseIndices(i) = featureIndices(featureIndexInSubspace)
          rawWeights(i) = randomState.nextGaussian()
          i += 1
        }

        // Compute the L2 norm of the vector
        var squaredSum = 0.0
        i = 0
        while (i < rawWeights.length) {
          squaredSum += rawWeights(i) * rawWeights(i)
          i += 1
        }
        val normValue = math.sqrt(squaredSum)

        // Zero-norm vector (astronomically unlikely with Gaussian draws) — emit leaf
        if (normValue == 0) {
          ExtendedExternalNode(numInstances)
        } else {
          // Normalize the vector and convert to float. Float precision is sufficient
          // for random hyperplane directions (features are already float), and the
          // double-precision offset handles the "where to split" precision — mirroring
          // how standard IF uses float features with a double splitValue.
          val normalizedWeights = new Array[Float](rawWeights.length)
          i = 0
          while (i < rawWeights.length) {
            normalizedWeights(i) = (rawWeights(i) / normValue).toFloat
            i += 1
          }

          // Sample intercept point p from the per-coordinate ranges.
          // Only coordinates with non-zero weights in the hyperplane matter.
          // The offset is computed with the float weights so that training and
          // scoring are consistent.
          var splitOffset = 0.0
          var k = 0
          while (k < sparseIndices.length) {
            val j = sparseIndices(k)
            var mn = Double.PositiveInfinity
            var mx = Double.NegativeInfinity
            var r = 0
            while (r < data.length) {
              val v = data(r).features(j).toDouble
              if (v < mn) mn = v
              if (v > mx) mx = v
              r += 1
            }
            val interceptValue = if (mn == mx) mn else mn + randomState.nextDouble() * (mx - mn)
            splitOffset += normalizedWeights(k) * interceptValue
            k += 1
          }

          // Canonicalize storage order for stable persistence and easier debugging.
          val sortedCoords = sparseIndices
            .zip(normalizedWeights)
            .sortBy { case (index, _) => index }
          val canonicalIndices = sortedCoords.map(_._1)
          val canonicalWeights = sortedCoords.map(_._2)
          val splitHyperplane =
            SplitHyperplane(canonicalIndices, canonicalWeights, splitOffset)

          // Partition: reference implementation uses (x - p) · n < 0 for the left branch,
          // which is equivalent to x·n < p·n (= splitOffset).
          val (leftData, rightData) = data.partition { point =>
            splitHyperplane.dot(point) < splitOffset
          }

          // No retry on degenerate splits — matching EIF paper Algorithm 2 and reference
          // implementation. Empty sides become ExtendedExternalNode(0), which contributes
          // avgPathLength(0) = 0.0 to the path length computation.
          val leftChild = generateExtendedIsolationTreeInternal(
            leftData,
            currentTreeHeight + 1,
            heightLimit,
            randomState,
            featureIndices,
            extensionLevel,
          )
          val rightChild = generateExtendedIsolationTreeInternal(
            rightData,
            currentTreeHeight + 1,
            heightLimit,
            randomState,
            featureIndices,
            extensionLevel,
          )
          ExtendedInternalNode(
            leftChild,
            rightChild,
            splitHyperplane,
          )
        }
      }
    }

    generateExtendedIsolationTreeInternal(
      data,
      0,
      heightLimit,
      randomState,
      featureIndices,
      extensionLevel,
    )
  }

  /**
   * Returns the path length from the root node of an extended isolation tree to the node in the
   * tree that contains a particular data point.
   *
   * @param dataInstance
   *   A single data point for scoring.
   * @param extendedNode
   *   The root node of the tree used to calculate the path length.
   * @return
   *   The path length to the instance.
   */
  def pathLength(dataInstance: DataPoint, extendedNode: ExtendedNode): Float = {

    /**
     * This recursive method returns the path length from a node of an isolation tree to the node in
     * the tree that contains a particular data point. The returned path length includes an
     * additional component dependent upon how many training data points ended up in this node.
     *
     * @param dataInstance
     *   A single data point for scoring.
     * @param node
     *   The root node of the tree used to calculate the path length.
     * @param currentPathLength
     *   The path length to the current node.
     * @return
     *   The path length to the instance.
     */
    @tailrec
    def pathLengthInternal(
      dataInstance: DataPoint,
      extendedNode: ExtendedNode,
      currentPathLength: Float,
    ): Float =

      extendedNode match {
        case ExtendedExternalNode(numInstances) =>
          currentPathLength + Utils.avgPathLength(numInstances)
        case ExtendedInternalNode(left, right, splitHyperplane) =>
          val dpVal = splitHyperplane.dot(dataInstance)
          // Match reference implementation: (x - p) · n < 0 ⇒ x·n < p·n (= offset) goes left.
          if (dpVal < splitHyperplane.offset) {
            pathLengthInternal(dataInstance, left, currentPathLength + 1)
          } else {
            pathLengthInternal(dataInstance, right, currentPathLength + 1)
          }
      }

    pathLengthInternal(dataInstance, extendedNode, 0.0f)
  }

  /**
   * Returns the path length from the root node of an extended isolation tree to the node in the
   * tree that contains a particular Spark vector.
   *
   * @param features
   *   A single data point for scoring.
   * @param extendedNode
   *   The root node of the tree used to calculate the path length.
   * @return
   *   The path length to the instance.
   */
  def pathLength(features: Vector, extendedNode: ExtendedNode): Float = {

    @tailrec
    def pathLengthInternal(
      features: Vector,
      extendedNode: ExtendedNode,
      currentPathLength: Float,
    ): Float =

      extendedNode match {
        case ExtendedExternalNode(numInstances) =>
          currentPathLength + Utils.avgPathLength(numInstances)
        case ExtendedInternalNode(left, right, splitHyperplane) =>
          val dpVal = splitHyperplane.dot(features)
          if (dpVal < splitHyperplane.offset) {
            pathLengthInternal(features, left, currentPathLength + 1)
          } else {
            pathLengthInternal(features, right, currentPathLength + 1)
          }
      }

    pathLengthInternal(features, extendedNode, 0.0f)
  }
}
