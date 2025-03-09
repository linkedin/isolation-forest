package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.NodesBase.{ExternalNodeBase, InternalNodeBase, NodeBase}

/**
 * Contains the node classes used to construct extended isolation trees.
 * An extended isolation tree splits data using a random hyperplane,
 * defined by (splitVector, splitOffset).
 */
private[isolationforest] case object ExtendedNodes {

  /**
   * Base trait for nodes in an extended isolation tree.
   * Subclasses are ExtendedExternalNode or ExtendedInternalNode.
   */
  sealed trait ExtendedNode extends NodeBase

  /**
   * An external (leaf) node in an extended isolation tree.
   *
   * @param numInstances The number of data points that terminated here.
   */
  case class ExtendedExternalNode(override val numInstances: Long)
    extends ExtendedNode with ExternalNodeBase {

    require(numInstances > 0, s"parameter numInstances must be > 0, but given invalid value $numInstances")
  }

  /**
   * An internal node in an extended isolation tree.
   * It splits data points by computing a dot product with splitVector,
   * then comparing it to splitOffset.
   *
   * @param splitVector The vector defining the hyperplane that splits the data at this node.
   * @param splitOffset The offset for the hyperplane.
   * @param leftChild   The left child node (data points with dot < splitOffset).
   * @param rightChild  The right child node (data points with dot >= splitOffset).
   */
  case class ExtendedInternalNode(
    leftChild: ExtendedNode,
    rightChild: ExtendedNode,
    splitVector: Array[Double],
    splitOffset: Double) extends ExtendedNode with InternalNodeBase {

    // Depth is 1 + max of children
    override val subtreeDepth: Int = 1 + math.max(leftChild.subtreeDepth, rightChild.subtreeDepth)

    require(splitVector.nonEmpty, "splitVector must be non-empty for an internal node.")

    override def toString: String =
      s"ExtendedInternalNode(" +
        s"splitVector=${splitVector.mkString("Array(", ", ", ")")}," +
        s" splitOffset=$splitOffset," +
        s" leftChild=($leftChild), rightChild=($rightChild))"
  }
}
