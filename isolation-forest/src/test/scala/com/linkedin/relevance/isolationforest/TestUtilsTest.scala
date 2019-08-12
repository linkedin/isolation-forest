package com.linkedin.relevance.isolationforest

import TestUtils._
import org.testng.Assert
import org.testng.annotations.Test


class TestUtilsTest {

  @Test(description = "loadMammographyDataTest")
  def loadMammographyDataTest(): Unit = {

    val spark = getSparkSession
    val data = loadMammographyData(spark)

    Assert.assertEquals(data.count(), 11183L)

    spark.stop()
  }

  @Test(description = "loadShuttleDataTest")
  def loadShuttleDataTest(): Unit = {

    val spark = getSparkSession
    val data = loadShuttleData(spark)

    Assert.assertEquals(data.count(), 49097L)

    spark.stop()
  }

  @Test(description = "readCsvTest")
  def readCsvTest(): Unit = {

    val data = readCsv("src/test/resources/shuttle.csv")

    Assert.assertEquals(data.length, 49097L)
  }
}
