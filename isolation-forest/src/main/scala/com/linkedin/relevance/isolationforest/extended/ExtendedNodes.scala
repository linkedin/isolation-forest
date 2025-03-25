package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import com.linkedin.relevance.isolationforest.core.NodesBase.{
  ExternalNodeBase,
  InternalNodeBase,
  NodeBase,
}

/**
 * Contains the node classes used to construct extended isolation trees. An extended isolation tree
 * splits data using a random hyperplane, defined by SplitHyperplane.
 */
private[isolationforest] case object ExtendedNodes {

  /**
   * Base trait for nodes in an extended isolation tree. Subclasses are ExtendedExternalNode or
   * ExtendedInternalNode.
   */
  sealed trait ExtendedNode extends NodeBase

  /**
   * An external (leaf) node in an extended isolation tree.
   *
   * @param numInstances
   *   The number of data points that terminated here.
   */
  case class ExtendedExternalNode(override val numInstances: Long)
      extends ExtendedNode
      with ExternalNodeBase {

    require(
      numInstances > 0,
      s"parameter numInstances must be > 0, but given invalid value $numInstances",
    )

    override def toString: String = s"ExtendedExternalNode(numInstances = $numInstances)"
  }

  /**
   * An internal node in an extended isolation tree.
   *
   * @param leftChild
   *   The left child node (data points with dot < splitOffset).
   * @param rightChild
   *   The right child node (data points with dot >= splitOffset).
   * @param splitHyperplane
   *   The norm vector and offset defining the hyperplane that splits the data at this node.
   */
  case class ExtendedInternalNode(
    leftChild: ExtendedNode,
    rightChild: ExtendedNode,
    splitHyperplane: SplitHyperplane,
  ) extends ExtendedNode
      with InternalNodeBase {

    // Depth is 1 + max of children
    override val subtreeDepth: Int = 1 + math.max(leftChild.subtreeDepth, rightChild.subtreeDepth)
    override def toString: String =
      s"ExtendedInternalNode(" +
        s"splitHyperplane=${splitHyperplane}" +
        s" leftChild=($leftChild), rightChild=($rightChild))"
  }
}
