package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.TestUtils._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._
import org.testng.Assert
import org.testng.annotations.Test

import java.io.File


class ExtendedIsolationForestTest {

  @Test(description = "extendedIsolationForestMammographyDataTest")
  def extendedIsolationForestMammographyDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val contamination = 0.02
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setContaminationError(contamination * 0.01)
      .setExtensionLevel(5)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val extendedIsolationForestModel = extendedIsolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = extendedIsolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    val aurocExpectation = 0.86
    val uncert = 0.02
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc === aurocExpectation +- uncert, "expected area under ROC =" +
      s" $aurocExpectation +/- $uncert, but observed $auroc")

    spark.stop()
  }

  @Test(description = "extendedIsolationForestMammographyDataTest")
  def extendedIsolationForestMammographyZeroExtensionDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val contamination = 0.02
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setContaminationError(contamination * 0.01)
      .setExtensionLevel(0)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val extendedIsolationForestModel = extendedIsolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = extendedIsolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocExpectation = 0.86
    val uncert = 0.02
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc === aurocExpectation +- uncert, "expected area under ROC =" +
      s" $aurocExpectation +/- $uncert, but observed $auroc")

    spark.stop()
  }

  @Test(description = "extnededIsolationForestMammographyExactContaminationDataTest")
  def extendedIsolationForestMammographyExactContaminationDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setContaminationError(0.0)
      .setExtensionLevel(5)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val extendedIsolationForestModel = extendedIsolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = extendedIsolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocExpectation = 0.87
    val uncert = 0.02
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc === aurocExpectation +- uncert, "expected area under ROC =" +
      s" $aurocExpectation +/- $uncert, but observed $auroc")

    spark.stop()
  }

  @Test(description = "extendedIsolationForestZeroContaminationTest")
  def extendedIsolationForestZeroContaminationTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new extended isolation forest model
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setExtensionLevel(5)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val isolationForestModel = extendedIsolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val predictedLabels = scores.map(x => x.predictedLabel).collect()
    val expectedLabels = Array.fill[Double](predictedLabels.length)(0.0)

    Assert.assertEquals(
      predictedLabels.toSeq,
      expectedLabels.toSeq,
      "expected all predicted labels to be 0.0")

    spark.stop()
  }

  @Test(description = "extendedIsolationForestShuttleDataTest")
  def extendedIsolationForestShuttleDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadShuttleData(spark)

    // Train a new isolation forest model
    val contamination = 0.07
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(contamination)
      .setContaminationError(contamination * 0.01)
      .setExtensionLevel(8)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val extendedIsolationForestModel = extendedIsolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = extendedIsolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    val aurocThreshold = 0.99
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc > aurocThreshold, s"Expected area under ROC > $aurocThreshold, but" +
      s" observed $auroc")

    spark.stop()
  }
}
