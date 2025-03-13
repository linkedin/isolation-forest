package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.extended.ExtendedNodes.{ExtendedExternalNode, ExtendedInternalNode}
import com.linkedin.relevance.isolationforest.core.TestUtils.readCsv
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import com.linkedin.relevance.isolationforest.extended.ExtendedUtils.SplitHyperplane
import org.testng.Assert
import org.testng.annotations.Test


class ExtendedIsolationTreeTest {

  @Test(description = "generateExtendedIsolationTreeTest")
  def generateExtendedIsolationTreeTest(): Unit = {

    val data = readCsv("src/test/resources/shuttle.csv")

    val dataArray = data.map(x => DataPoint(x.slice(0, data.head.length - 1)))  // Drop labels column

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
}
