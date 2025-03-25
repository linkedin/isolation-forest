package com.linkedin.relevance.isolationforest.core

/**
 * A base trait for an isolation tree. Both "standard" and "extended" trees can extend this trait to
 * have a consistent interface.
 */
private[isolationforest] trait IsolationTreeBase extends Serializable {

  /**
   * Calculate the path length for a given data point in this tree.
   *
   * :param dataPoint: The data point for which to calculate the path length. :return: The path
   * length for the data point.
   */
  def calculatePathLength(dataPoint: Utils.DataPoint): Float
}
