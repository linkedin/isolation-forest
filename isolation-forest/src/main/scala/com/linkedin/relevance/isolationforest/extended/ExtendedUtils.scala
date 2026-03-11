package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import org.apache.spark.ml.linalg.Vector

/**
 * Useful utilities for extended isolation forest.
 */
private[isolationforest] object ExtendedUtils extends Serializable {

  /**
   * Represents a hyperplane that splits the data.
   *
   * @param indices
   *   Global feature indices for the non-zero coordinates of the hyperplane normal.
   * @param weights
   *   Non-zero coordinates of the normalized hyperplane normal.
   * @param offset
   *   The offset of the hyperplane.
   */
  final class SplitHyperplane private (
    val indices: Array[Int],
    val weights: Array[Double],
    val offset: Double,
  ) extends Serializable {

    require(indices.nonEmpty, "indices must be non-empty.")
    require(
      indices.length == weights.length,
      "indices and weights must have the same length.",
    )
    require(indices.forall(_ >= 0), "indices must be non-negative.")
    require(indices.distinct.length == indices.length, "indices must be distinct.")
    require(indices.sameElements(indices.sorted), "indices must be sorted in ascending order.")

    def dot(point: DataPoint): Double = {
      var sum = 0.0
      var i = 0
      while (i < indices.length) {
        sum += weights(i) * point.features(indices(i))
        i += 1
      }
      sum
    }

    def dot(features: Vector): Double = {
      var sum = 0.0
      var i = 0
      while (i < indices.length) {
        sum += weights(i) * features(indices(i)).toFloat
        i += 1
      }
      sum
    }

    override def toString: String =
      s"SplitHyperplane(" +
        s"indices = (${indices.mkString(", ")}), " +
        s"weights = (${weights.mkString(", ")}), " +
        s"offset = $offset)"
  }

  object SplitHyperplane {

    def apply(indices: Array[Int], weights: Array[Double], offset: Double): SplitHyperplane =
      new SplitHyperplane(indices, weights, offset)
  }
}
