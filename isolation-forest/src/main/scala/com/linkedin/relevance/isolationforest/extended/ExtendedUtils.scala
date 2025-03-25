package com.linkedin.relevance.isolationforest.extended

/**
 * Useful utilities for extended isolation forest.
 */
private[isolationforest] object ExtendedUtils extends Serializable {

  /**
   * Represents a hyperplane that splits the data.
   *
   * @param norm
   *   The normal vector of the hyperplane.
   * @param offset
   *   The offset of the hyperplane.
   */
  case class SplitHyperplane(norm: Array[Double], offset: Double) {
    require(norm.nonEmpty, "splitVector must be non-empty.")

    override def toString: String =
      s"SplitHyperplane(norm = (${norm.mkString(", ")}), offset = $offset)"
  }
}
