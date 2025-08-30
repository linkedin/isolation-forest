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
   *   coordinates. 0 => standard iForest splits, dimensionOfSubspace-1 => fully extended splits"
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
     *   coordinates. 0 => standard iForest splits, dimensionOfSubspace-1 => fully extended splits"
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
      val numFeatures = data.head.features.length

      if (currentTreeHeight >= heightLimit || numInstances <= 1) {
        // Leaf node
        ExtendedExternalNode(numInstances)
      } else {
        val dim = featureIndices.length
        // We allow up to (extensionLevel+1) coordinates to be non-zero
        val nNonZero = math.min(extensionLevel + 1, dim)

        // Build the hyperplane norm vector for the full space
        val splitVector = Array.fill(numFeatures)(0.0)

        // Randomly choose which feature indices become non-zero (in the selected subspace)
        val chosenFeatureIndices = randomState.shuffle((0 until dim).toList).take(nNonZero)
        chosenFeatureIndices.foreach { i =>
          splitVector(featureIndices(i)) = randomState.nextGaussian()
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

        // Compute dot products in this subspace for a quick degeneracy check
        var minDot = Double.PositiveInfinity
        var maxDot = Double.NegativeInfinity
        var idx = 0
        while (idx < data.length) {
          val d = dot(normSplitVector, data(idx))
          if (d < minDot) minDot = d
          if (d > maxDot) maxDot = d
          idx += 1
        }

        // If all dot-values are the same => leaf
        if (minDot == maxDot) {
          ExtendedExternalNode(numInstances)
        } else {
          // -------- FIX: sample intercept as a point p from the per-coordinate ranges --------
          // Only coordinates with non-zero weights in the hyperplane matter.
          val nonZeroFeatureIdxs: Array[Int] = chosenFeatureIndices.map(featureIndices).toArray
          val p = Array.fill(numFeatures)(0.0)

          var k = 0
          while (k < nonZeroFeatureIdxs.length) {
            val j = nonZeroFeatureIdxs(k)
            var mn = Double.PositiveInfinity
            var mx = Double.NegativeInfinity
            var r = 0
            while (r < data.length) {
              val v = data(r).features(j).toDouble
              if (v < mn) mn = v
              if (v > mx) mx = v
              r += 1
            }
            // If this coordinate is constant at this node, just use that constant.
            p(j) = if (mn == mx) mn else mn + randomState.nextDouble() * (mx - mn)
            k += 1
          }

          // Offset = n · p
          var splitOffset = 0.0
          var t = 0
          while (t < numFeatures) { splitOffset += normSplitVector(t) * p(t); t += 1 }
          // -----------------------------------------------------------------------------------

          // Partition into left (dot < offset) and right (>= offset).
          // Paper's Algorithms 2 & 3 use (x - p) · n ≤ 0 for the left branch,
          // which is equivalent to x·n ≤ p·n (= splitOffset).
          val (leftData, rightData) = data.partition { point =>
            dot(normSplitVector, point) <= splitOffset
          }

          // If one side is empty, treat as leaf
          if (leftData.isEmpty || rightData.isEmpty) {
            ExtendedExternalNode(numInstances)
          } else {
            // Build children
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
              SplitHyperplane(normSplitVector, splitOffset),
            )
          }
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
          val dpVal = dot(splitHyperplane.norm, dataInstance)
          // Match Algorithm 3’s test: (x - p) · n ≤ 0 ⇒ x·n ≤ p·n (= offset) goes left.
          if (dpVal <= splitHyperplane.offset) {
            pathLengthInternal(dataInstance, left, currentPathLength + 1)
          } else {
            pathLengthInternal(dataInstance, right, currentPathLength + 1)
          }
      }

    pathLengthInternal(dataInstance, extendedNode, 0.0f)
  }

  /**
   * Compute x · n where `splitVector` is the *full-length* normal (zeros in coordinates not used by
   * this node), matching the paper's test (x - p) · n ≤ 0.
   *
   * @param splitVector
   *   array of length == point.features.length
   * @param point
   *   data point (float[] features)
   * @return
   *   dot product
   */
  def dot(splitVector: Array[Double], point: DataPoint): Double = {
    var sum = 0.0
    var i = 0
    while (i < splitVector.length) {
      sum += splitVector(i) * point.features(i)
      i += 1
    }
    sum
  }
}
