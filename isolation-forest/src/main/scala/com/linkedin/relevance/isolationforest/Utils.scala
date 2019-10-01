package com.linkedin.relevance.isolationforest


/**
  * Useful utilities.
  */
private[isolationforest] object Utils extends Serializable {

  case class DataPoint(features: Array[Float])
  case class OutlierScore(score: Double)

  val EulerConstant = 0.5772156649f

  /**
    * Returns the average path length for an unsuccessful BST search. It is Equation 1 in the 2008
    * "Isolation Forest" paper by F. T. Liu, et al.
    *
    * @param numInstances The number of data points in the root node of the BST.
    * @return The average path length of an unsuccessful BST search.
    */
  def avgPathLength(numInstances: Long): Float = {

    if (numInstances <= 1) {
      0.0f
    } else {
      2.0f * (math.log(numInstances.toFloat - 1.0f).toFloat + EulerConstant) -
        (2.0f * (numInstances.toFloat - 1.0f) / numInstances.toFloat)
    }
  }
}
