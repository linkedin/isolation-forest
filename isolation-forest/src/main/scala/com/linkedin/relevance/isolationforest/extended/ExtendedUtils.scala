package com.linkedin.relevance.isolationforest.extended

/**
 * Useful utilities for extended isolation forest.
 */
private[isolationforest] object ExtendedUtils extends Serializable {

  case class SplitHyperplane(norm: Array[Double], offset: Double) {
    require(norm.nonEmpty, "splitVector must be non-empty.")

    override def toString: String = s"SplitHyperplane(norm = (${norm.mkString(", ")}), offset = $offset)"
  }
}
