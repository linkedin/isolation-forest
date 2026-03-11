package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.extended.ExtendedNodes.{
  ExtendedExternalNode,
  ExtendedInternalNode,
}
import com.linkedin.relevance.isolationforest.core.TestUtils.readCsv
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import org.testng.Assert
import org.testng.annotations.Test

class ExtendedIsolationTreeTest {

  @Test(description = "generateExtendedIsolationTreeTest")
  def generateExtendedIsolationTreeTest(): Unit = {

    val data = readCsv("src/test/resources/shuttle.csv")

    val dataArray = data.map(x => DataPoint(x.slice(0, data.head.length - 1))) // Drop labels column

    val heightLimit = 15
    val randomState = new scala.util.Random(1)
    val featureIndices = dataArray.head.features.indices.toArray
    val root = ExtendedIsolationTree
      .generateExtendedIsolationTree(dataArray, heightLimit, randomState, featureIndices, 1)

    Assert.assertEquals(root.subtreeDepth, heightLimit)
  }

  @Test(description = "pathLengthTest")
  def pathLengthTest(): Unit = {

    val leftChild = ExtendedExternalNode(10)
    val rightChild = ExtendedExternalNode(20)
    val splitHyperplane = SplitHyperplane(Array(0.7071067812, 0.7071067812), 2.5)
    val root = ExtendedInternalNode(leftChild, rightChild, splitHyperplane)

    val data1 = DataPoint(Array(1.0f, 2.0f))
    val data2 = DataPoint(Array(2.0f, 3.0f))

    val pathLength1 = ExtendedIsolationTree.pathLength(data1, root)
    val pathLength2 = ExtendedIsolationTree.pathLength(data2, root)

    Assert.assertEquals(pathLength1, 4.7488804f)
    Assert.assertEquals(pathLength2, 6.143309f)
  }

  @Test(description = "zeroSizeLeafNodeTest")
  def zeroSizeLeafNodeTest(): Unit = {

    // ExtendedExternalNode(0) must be constructable — this is first-class EIF behavior
    // when a degenerate hyperplane split sends all points to one side.
    val zeroLeaf = ExtendedExternalNode(0)
    Assert.assertEquals(zeroLeaf.numInstances, 0L)
    Assert.assertEquals(zeroLeaf.subtreeDepth, 0)
  }

  @Test(description = "zeroSizeLeafPathLengthTest")
  def zeroSizeLeafPathLengthTest(): Unit = {

    // A zero-size leaf contributes avgPathLength(0) = 0.0 to the path length.
    // Build a tree: root splits left=zeroLeaf, right=normalLeaf.
    val zeroLeaf = ExtendedExternalNode(0)
    val normalLeaf = ExtendedExternalNode(5)
    val splitHyperplane = SplitHyperplane(Array(1.0, 0.0), 0.5)
    val root = ExtendedInternalNode(zeroLeaf, normalLeaf, splitHyperplane)

    // Point that goes left (dot = 0.0 < 0.5 offset) — hits the zero-size leaf
    val leftPoint = DataPoint(Array(0.0f, 1.0f))
    val pathLengthLeft = ExtendedIsolationTree.pathLength(leftPoint, root)
    // Expected: currentPathLength(1) + avgPathLength(0) = 1.0 + 0.0 = 1.0
    Assert.assertEquals(pathLengthLeft, 1.0f)

    // Point that goes right (dot = 1.0 >= 0.5 offset) — hits the normal leaf
    val rightPoint = DataPoint(Array(1.0f, 1.0f))
    val pathLengthRight = ExtendedIsolationTree.pathLength(rightPoint, root)
    // Expected: currentPathLength(1) + avgPathLength(5) > 1.0
    Assert.assertTrue(pathLengthRight > 1.0f, "path through non-zero leaf should exceed 1.0")
  }

  @Test(description = "hyperplaneNormalsAreL2NormalizedTest")
  def hyperplaneNormalsAreL2NormalizedTest(): Unit = {

    // Every internal node's norm vector should have L2 norm = 1.0
    val data = Array(
      DataPoint(Array(1.0f, 2.0f, 3.0f)),
      DataPoint(Array(4.0f, 5.0f, 6.0f)),
      DataPoint(Array(7.0f, 8.0f, 9.0f)),
      DataPoint(Array(2.0f, 3.0f, 1.0f)),
      DataPoint(Array(5.0f, 1.0f, 4.0f)),
      DataPoint(Array(8.0f, 6.0f, 2.0f)),
      DataPoint(Array(3.0f, 9.0f, 7.0f)),
      DataPoint(Array(6.0f, 4.0f, 8.0f)),
    )

    val featureIndices = Array(0, 1, 2)

    // Test across multiple extension levels and seeds
    for {
      extLevel <- 0 to 2
      seed <- Seq(1L, 42L, 123L)
    } {
      val randomState = new scala.util.Random(seed)
      val root = ExtendedIsolationTree.generateExtendedIsolationTree(
        data,
        heightLimit = 4,
        randomState,
        featureIndices,
        extensionLevel = extLevel,
      )

      def assertNormalized(node: ExtendedNodes.ExtendedNode): Unit = node match {
        case ExtendedInternalNode(left, right, hp) =>
          val l2Norm = math.sqrt(hp.norm.map(x => x * x).sum)
          Assert.assertEquals(
            l2Norm,
            1.0,
            1e-10,
            s"Hyperplane norm should be L2-normalized (seed=$seed, extLevel=$extLevel)," +
              s" but L2 norm was $l2Norm",
          )
          assertNormalized(left)
          assertNormalized(right)
        case _: ExtendedExternalNode => // leaf, nothing to check
      }

      assertNormalized(root)
    }
  }

  @Test(description = "extensionLevelZeroProducesAxisAlignedSplitsTest")
  def extensionLevelZeroProducesAxisAlignedSplitsTest(): Unit = {

    // With extensionLevel=0, nNonZero = min(0+1, dim) = 1, so every hyperplane
    // normal vector should have exactly one non-zero coordinate (axis-aligned split).
    val data = Array(
      DataPoint(Array(1.0f, 2.0f, 3.0f)),
      DataPoint(Array(4.0f, 5.0f, 6.0f)),
      DataPoint(Array(7.0f, 8.0f, 9.0f)),
      DataPoint(Array(2.0f, 3.0f, 1.0f)),
      DataPoint(Array(5.0f, 1.0f, 4.0f)),
      DataPoint(Array(8.0f, 6.0f, 2.0f)),
      DataPoint(Array(3.0f, 9.0f, 7.0f)),
      DataPoint(Array(6.0f, 4.0f, 8.0f)),
    )

    val heightLimit = 4
    val randomState = new scala.util.Random(42)
    val featureIndices = Array(0, 1, 2)
    val root = ExtendedIsolationTree.generateExtendedIsolationTree(
      data,
      heightLimit,
      randomState,
      featureIndices,
      extensionLevel = 0,
    )

    // Walk the tree and check that every internal node's norm has exactly 1 non-zero entry
    def assertAxisAligned(node: ExtendedNodes.ExtendedNode): Unit = node match {
      case ExtendedInternalNode(left, right, hp) =>
        val nonZeroCount = hp.norm.count(_ != 0.0)
        Assert.assertEquals(
          nonZeroCount,
          1,
          s"extensionLevel=0 should produce axis-aligned splits, but got $nonZeroCount non-zero entries",
        )
        assertAxisAligned(left)
        assertAxisAligned(right)
      case _: ExtendedExternalNode => // leaf, nothing to check
    }

    assertAxisAligned(root)
  }
}
