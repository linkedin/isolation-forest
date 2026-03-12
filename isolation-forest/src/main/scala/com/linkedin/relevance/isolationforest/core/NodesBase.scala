package com.linkedin.relevance.isolationforest.core

private[isolationforest] case object NodesBase {

  /**
   * A minimal trait capturing what every "node" must define. Both standard and extended trees will
   * implement this trait.
   */
  trait NodeBase extends Serializable {

    /** The subtree depth: 0 for a leaf, otherwise 1 + max of children. */
    val subtreeDepth: Int
    def toString: String
  }

  /**
   * A trait capturing a "leaf node" concept: typically has a count of how many training points
   * landed here.
   */
  trait ExternalNodeBase extends NodeBase {

    val numInstances: Long

    override val subtreeDepth: Int = 0
  }

  /**
   * A trait capturing an "internal node" concept: typically has leftChild, rightChild, plus a
   * subtreeDepth that depends on them.
   *
   * Note that we do NOT finalize the fields here, because standard vs extended might have different
   * node types.
   */
  trait InternalNodeBase extends NodeBase {

    def leftChild: NodeBase
    def rightChild: NodeBase
  }
}
