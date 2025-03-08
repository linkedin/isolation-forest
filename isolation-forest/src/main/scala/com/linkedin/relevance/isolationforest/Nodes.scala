package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.NodesBase.{ExternalNodeBase, InternalNodeBase, NodeBase}

/**
  * Contains the node classes used to construct isolation trees.
  */
private[isolationforest] case object Nodes {

  /**
    * Base trait Nodes used in the isolation forest.
    */
  sealed trait Node extends NodeBase {
  }

  /**
    * External nodes in an isolation tree.
    *
    * @param numInstances The number of data points that terminated in this node during training.
    */
  case class ExternalNode(numInstances: Long) extends Node with ExternalNodeBase {

    require(numInstances > 0, "parameter numInstances must be >0, but given invalid value" +
      s" ${numInstances}")
  }

  /**
    * Internal nodes in an isolation tree.
    *
    * @param leftChild      The left child node produced by splitting the data points at this node.
    * @param rightChild     The right child node by splitting the data points at this node.
    * @param splitAttribute The feature upon which the split was performed.
    * @param splitValue     The feature value used for the split.
    */
  case class InternalNode(
    leftChild: Node,
    rightChild: Node,
    splitAttribute: Int,
    splitValue: Double) extends Node with InternalNodeBase {

    require(splitAttribute >= 0, "parameter splitAttribute must be >=0, but given invalid value" +
      s" ${splitAttribute}")

    override val subtreeDepth: Int = 1 + math.max(leftChild.subtreeDepth, rightChild.subtreeDepth)

    override def toString: String =
      s"InternalNode(splitAttribute = $splitAttribute, splitValue = $splitValue," +
        s" leftChild = ($leftChild), rightChild = ($rightChild))"
  }
}
