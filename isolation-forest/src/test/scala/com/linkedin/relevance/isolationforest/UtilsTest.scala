package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Utils.avgPathLength
import org.testng.Assert
import org.testng.annotations.Test


class UtilsTest {

  @Test(description = "avgPathLengthMethodTest")
  def avgPathLengthMethodTest(): Unit = {

    Assert.assertEquals(avgPathLength(0), 0.0f)
    Assert.assertEquals(avgPathLength(1), 0.0f)
    Assert.assertEquals(avgPathLength(2), 0.15443134f)
    Assert.assertEquals(avgPathLength(10), 3.7488806f)
    Assert.assertEquals(avgPathLength(Long.MaxValue), 86.49098f)
  }
}
