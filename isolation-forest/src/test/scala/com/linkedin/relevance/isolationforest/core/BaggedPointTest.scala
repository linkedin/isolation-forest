/*
 * Some of these tests were taken and modified from the spark.ml files BaggedPointSuite and
 * EnsembleTestHelper, which are open sourced under the Apache 2.0 license.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Functions based upon modified versions of those in spark.ml:
 * 1) generateDataPoints
 * 2) testRandomArrays
 * 3) baggedPointRDDWithoutSamplingTest
 * 4) baggedPointRDDWithSubsamplingWithReplacementFraction1p0Test
 * 5) baggedPointRDDWithSubsamplingWithReplacementFraction0p5Test
 * 6) baggedPointRDDWithSubsamplingWithoutReplacementFraction1p0Test
 * 7) baggedPointRDDWithSubsamplingWithoutReplacementFraction0p5Test
 *
 */

package com.linkedin.relevance.isolationforest.core

import TestUtils._
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import org.apache.spark.util.StatCounter
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._
import org.testng.Assert
import org.testng.annotations.Test

import scala.collection.mutable

class BaggedPointTest {

  def generateDataPoints(numFeatures: Int, numInstances: Int): Array[DataPoint] = {

    val dataPointArray = new Array[DataPoint](numInstances)
    for (i <- 0 until numInstances) {
      val features = Array.fill[Float](numFeatures)(i.toFloat)
      dataPointArray(i) = new DataPoint(features)
    }
    dataPointArray
  }

  def testRandomArrays(
    data: Array[Array[Double]],
    numCols: Int,
    expectedMean: Double,
    expectedStddev: Double,
    epsilon: Double,
  ): Unit = {

    val values = new mutable.ArrayBuffer[Double]()
    data.foreach { row =>
      assert(row.size == numCols)
      values ++= row.map(x => x)
    }
    val stats = new StatCounter(values)
    assert(math.abs(stats.mean - expectedMean) < epsilon)
    assert(math.abs(stats.stdev - expectedStddev) < epsilon)
  }

  @Test(description = "baggedPointRDDWithoutSamplingTest")
  def baggedPointRDDWithoutSamplingTest(): Unit = {

    val spark = getSparkSession

    val dataPointArray = generateDataPoints(1, 1000)
    val rdd = spark.sparkContext.parallelize(dataPointArray.toIndexedSeq)
    val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, 1, false, 42)
    baggedRDD.collect().foreach { baggedPoint =>
      assert(baggedPoint.subsampleWeights.size == 1 && baggedPoint.subsampleWeights(0) == 1)
    }
  }

  @Test(description = "baggedPointRDDWithSubsamplingWithReplacementFraction1p0Test")
  def baggedPointRDDWithSubsamplingWithReplacementFraction1p0Test(): Unit = {

    val spark = getSparkSession

    val numSubsamples = 100
    val (expectedMean, expectedStddev) = (1.0, 1.0)

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = generateDataPoints(1, 1000)
    val rdd = spark.sparkContext.parallelize(arr.toIndexedSeq)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, numSubsamples, true, seed)
      val subsampleCounts: Array[Array[Double]] = baggedRDD
        .map(_.subsampleWeights.map(x => x))
        .collect()
      testRandomArrays(subsampleCounts, numSubsamples, expectedMean, expectedStddev, epsilon = 0.01)
    }
  }

  @Test(description = "baggedPointRDDWithSubsamplingWithReplacementFraction0p5Test")
  def baggedPointRDDWithSubsamplingWithReplacementFraction0p5Test(): Unit = {

    val spark = getSparkSession

    val numSubsamples = 100
    val subsample = 0.5
    val (expectedMean, expectedStddev) = (subsample, math.sqrt(subsample))

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = generateDataPoints(1, 1000)
    val rdd = spark.sparkContext.parallelize(arr.toIndexedSeq)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, subsample, numSubsamples, true, seed)
      val subsampleCounts: Array[Array[Double]] = baggedRDD
        .map(_.subsampleWeights.map(x => x))
        .collect()
      testRandomArrays(subsampleCounts, numSubsamples, expectedMean, expectedStddev, epsilon = 0.01)
    }
  }

  @Test(description = "baggedPointRDDWithSubsamplingWithoutReplacementFraction1p0Test")
  def baggedPointRDDWithSubsamplingWithoutReplacementFraction1p0Test(): Unit = {

    val spark = getSparkSession

    val numSubsamples = 100
    val (expectedMean, expectedStddev) = (1.0, 0)

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = generateDataPoints(1, 1000)
    val rdd = spark.sparkContext.parallelize(arr.toIndexedSeq)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, 1.0, numSubsamples, false, seed)
      val subsampleCounts: Array[Array[Double]] = baggedRDD
        .map(_.subsampleWeights.map(x => x))
        .collect()
      testRandomArrays(subsampleCounts, numSubsamples, expectedMean, expectedStddev, epsilon = 0.01)
    }
  }

  @Test(description = "baggedPointRDDWithSubsamplingWithoutReplacementFraction0p5Test")
  def baggedPointRDDWithSubsamplingWithoutReplacementFraction0p5Test(): Unit = {

    val spark = getSparkSession

    val numSubsamples = 100
    val subsample = 0.5
    val (expectedMean, expectedStddev) = (subsample, math.sqrt(subsample * (1 - subsample)))

    val seeds = Array(123, 5354, 230, 349867, 23987)
    val arr = generateDataPoints(1, 1000)
    val rdd = spark.sparkContext.parallelize(arr.toIndexedSeq)
    seeds.foreach { seed =>
      val baggedRDD = BaggedPoint.convertToBaggedRDD(rdd, subsample, numSubsamples, false, seed)
      val subsampleCounts: Array[Array[Double]] = baggedRDD
        .map(_.subsampleWeights)
        .collect()
      testRandomArrays(subsampleCounts, numSubsamples, expectedMean, expectedStddev, epsilon = 0.01)
    }
  }

  @Test(description = "flattenBaggedRDDTest")
  def flattenBaggedRDDTest(): Unit = {

    val spark = getSparkSession

    val dataPointArray = generateDataPoints(10, 2)
    val subsampleWeights = Array(1.0, 3.0)
    val expectedResult = Array(
      (0, dataPointArray(0)),
      (0, dataPointArray(1)),
      (1, dataPointArray(0)),
      (1, dataPointArray(0)),
      (1, dataPointArray(0)),
      (1, dataPointArray(1)),
      (1, dataPointArray(1)),
      (1, dataPointArray(1)),
    )

    val dataPointRDD = spark.sparkContext.parallelize(dataPointArray.toIndexedSeq)
    val baggedPointRDD = dataPointRDD.map(x => new BaggedPoint(x, subsampleWeights))
    val flattenedBaggedPointRDD = BaggedPoint.flattenBaggedRDD(baggedPointRDD, 1L)
    val flattenedBaggedPointArray = flattenedBaggedPointRDD.collect()

    val expectedSumArray = expectedResult.map(x => x._1 + x._2.features.sum).sorted
    val actualSumArray = flattenedBaggedPointArray.map(x => x._1 + x._2.features.sum).sorted
    Assert.assertEquals(expectedSumArray.toSeq, actualSumArray.toSeq)
  }

  @Test(description = "flattenBaggedRDDNonIntegerWeightTest")
  def flattenBaggedRDDNonIntegerWeightTest(): Unit = {

    val spark = getSparkSession

    val numRecords = 10000
    val dataPointArray = generateDataPoints(10, numRecords)
    val subsampleWeights = Array(1.3, 1.75)

    val dataPointRDD = spark.sparkContext.parallelize(dataPointArray.toIndexedSeq)
    val baggedPointRDD = dataPointRDD.map(x => new BaggedPoint(x, subsampleWeights))
    val flattenedBaggedPointRDD = BaggedPoint.flattenBaggedRDD(baggedPointRDD, 1L)
    val flattenedBaggedPointArray = flattenedBaggedPointRDD.collect()

    val tol = 0.05

    val result1 = flattenedBaggedPointArray.count(x => x._1 == 0).toDouble
    Assert.assertTrue(
      result1 === subsampleWeights(0) * numRecords +- numRecords * tol,
      s"expected ${subsampleWeights(0) * numRecords} +/- ${numRecords * tol}, but observed" +
        s" ${result1}",
    )

    val result2 = flattenedBaggedPointArray.count(x => x._1 == 1).toDouble
    Assert.assertTrue(
      result2 === subsampleWeights(1) * numRecords +- numRecords * tol,
      s"expected ${subsampleWeights(1) * numRecords} +/- ${numRecords * tol}, but observed" +
        s" ${result2}",
    )
  }

  @Test(description = "baggedPointEmptyRDDTest")
  def baggedPointEmptyRDDTest(): Unit = {
    val spark = getSparkSession
    // Create an empty RDD of DataPoint
    val emptyRDD = spark.sparkContext.emptyRDD[DataPoint]

    // Attempt to create a bagged RDD from an empty input
    val baggedRDD = BaggedPoint.convertToBaggedRDD(emptyRDD, 0.5, 5, withReplacement = true, 42L)

    // We expect no output
    Assert.assertEquals(baggedRDD.count(), 0L, "Bagged RDD should be empty when input is empty.")
  }

  @Test(description = "baggedPointSingleRecordTest")
  def baggedPointSingleRecordTest(): Unit = {
    val spark = getSparkSession
    // Generate exactly one DataPoint
    val singleDataPoint = generateDataPoints(numFeatures = 3, numInstances = 1)
    val rdd = spark.sparkContext.parallelize(singleDataPoint.toIndexedSeq, numSlices = 1)

    // Subsample with a small rate, multiple subsamples, no replacement
    val baggedRDD = BaggedPoint.convertToBaggedRDD(
      rdd,
      subsamplingRate = 0.5,
      numSubsamples = 3,
      withReplacement = false,
      seed = 100L,
    )

    // There should be exactly one BaggedPoint in the result
    Assert.assertEquals(
      baggedRDD.count(),
      1L,
      "Expected exactly 1 bagged point for single-record RDD.",
    )

    // Flatten and see how many copies appear
    val flattened = BaggedPoint.flattenBaggedRDD(baggedRDD, seed = 100L).collect()
    // Because it's Binomial(1, 0.5) for 3 subsamples, we expect an integer between 0 and 3 total.
    Assert.assertTrue(
      flattened.length >= 0 && flattened.length <= 3,
      s"Flattened length should be between 0 and 3, found ${flattened.length}.",
    )
  }

  @Test(description = "baggedPointVerySmallSubsamplingRateTest")
  def baggedPointVerySmallSubsamplingRateTest(): Unit = {
    val spark = getSparkSession
    // Generate some data
    val data = generateDataPoints(numFeatures = 1, numInstances = 2000)
    val rdd = spark.sparkContext.parallelize(data.toIndexedSeq)

    // Use a very small subsampling rate
    val verySmallRate = 0.001
    val baggedRDD =
      BaggedPoint.convertToBaggedRDD(
        rdd,
        verySmallRate,
        numSubsamples = 5,
        withReplacement = true,
        seed = 9999L,
      )

    // Collect subsample weights
    val subsampleWeights: Array[Array[Double]] = baggedRDD.map(_.subsampleWeights).collect()

    // Optional: We can do a rough check that the average is near the Poisson mean of 0.001 for each subsample
    // Reuse your existing testRandomArrays if you like, or do a simpler sanity check:
    val meanWeights = subsampleWeights.flatten.sum / subsampleWeights.flatten.length
    Assert.assertTrue(
      math.abs(meanWeights - verySmallRate) < 0.0005,
      s"Mean weight $meanWeights should be near $verySmallRate for a very small sampling rate.",
    )
  }

  @Test(description = "baggedPointConsistencyOfResultsTest")
  def baggedPointConsistencyOfResultsTest(): Unit = {
    val spark = getSparkSession
    val data = generateDataPoints(numFeatures = 2, numInstances = 100)
    val rdd = spark.sparkContext.parallelize(data.toIndexedSeq)

    // We run convertToBaggedRDD twice with the same seed and compare
    val seed = 2020L
    val baggedRDD1 =
      BaggedPoint.convertToBaggedRDD(
        rdd,
        subsamplingRate = 0.3,
        numSubsamples = 5,
        withReplacement = true,
        seed,
      )
    val baggedRDD2 =
      BaggedPoint.convertToBaggedRDD(
        rdd,
        subsamplingRate = 0.3,
        numSubsamples = 5,
        withReplacement = true,
        seed,
      )

    // Collect the results
    val arr1 = baggedRDD1.map(_.subsampleWeights).collect()
    val arr2 = baggedRDD2.map(_.subsampleWeights).collect()

    // They should be the same length
    Assert.assertEquals(
      arr1.length,
      arr2.length,
      "Both runs should produce the same number of BaggedPoints.",
    )

    // Compare them element-wise to ensure they match
    // (We expect EXACT match because same seed => same distribution draws)
    for (i <- arr1.indices)
      Assert.assertEquals(
        arr1(i),
        arr2(i),
        s"Mismatch in subsampleWeights for record $i with the same seed = $seed.",
      )
  }
}
