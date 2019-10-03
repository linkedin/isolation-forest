package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.TestUtils._
import java.io.File
import org.apache.commons.io.FileUtils.deleteDirectory
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._
import org.testng.Assert
import org.testng.annotations.Test


class IsolationForestTest {

  @Test(description = "isolationForestEstimatorWriteReadTest")
  def isolationForestEstimatorWriteReadTest(): Unit = {

    val spark = getSparkSession

    val savePath = System.getProperty("java.io.tmpdir") + "/isolationForestEstimatorWriteReadTest"

    val contamination = 0.02
    val isolationForest1 = new IsolationForest()
      .setNumEstimators(200)
      .setBootstrap(true)
      .setMaxSamples(10000)
      .setMaxFeatures(0.7)
      .setFeaturesCol("featuresTestColumn")
      .setPredictionCol("predictedLabelTestColumn")
      .setScoreCol("outlierScoreTestColumn")
      .setContamination(contamination)
      .setContaminationError(contamination * 0.01)
      .setRandomSeed(1)

    isolationForest1.write.overwrite.save(savePath)
    val isolationForest2 = IsolationForest.load(savePath)
    deleteDirectory(new File(savePath))

    Assert.assertEquals(
      isolationForest1.extractParamMap.toString,
      isolationForest2.extractParamMap.toString)

    spark.stop()
  }

  @Test(description = "isolationForestMammographyDataTest")
  def isolationForestMammographyDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val contamination = 0.02
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setContaminationError(contamination * 0.01)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val isolationForestModel = isolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocExpectation = 0.86
    val uncert = 0.02
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc === aurocExpectation +- uncert, "expected area under ROC =" +
      s" $aurocExpectation +/- $uncert, but observed $auroc")

    spark.stop()
  }

  @Test(description = "isolationForestMammographyExactContaminationDataTest")
  def isolationForestMammographyExactContaminationDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setContaminationError(0.0)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val isolationForestModel = isolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocExpectation = 0.86
    val uncert = 0.02
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc === aurocExpectation +- uncert, "expected area under ROC =" +
      s" $aurocExpectation +/- $uncert, but observed $auroc")

    spark.stop()
  }

  @Test(description = "isolationForestZeroContaminationTest")
  def isolationForestZeroContaminationTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new isolation forest model
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val isolationForestModel = isolationForest.fit(data)

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val predictedLabels = scores.map(x => x.predictedLabel).collect
    val expectedLabels = Array.fill[Double](predictedLabels.length)(0.0)
    Assert.assertEquals(
      predictedLabels.deep,
      expectedLabels.deep,
      "expected all predicted labels to be 0.0")

    spark.stop()
  }

  @Test(description = "isolationForestShuttleDataTest")
  def isolationForestShuttleDataTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadShuttleData(spark)

    // Train a new isolation forest model
    val contamination = 0.07
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(contamination)
      .setContaminationError(contamination * 0.01)
      .setRandomSeed(1)

    // Score all training data instances using the new model
    val isolationForestModel = isolationForest.fit(data)

    // Calculate the mean of the labeled inlier and outlier points and assert
    val labeledOutlierScores = isolationForestModel
      .transform(data.filter(_.label == 1.0))
      .as[ScoringResult]
    val labeledInlierScores = isolationForestModel
      .transform(data.filter(_.label == 0.0))
      .as[ScoringResult]

    val labeledOutlierScoresMean = labeledOutlierScores
      .map(_.outlierScore)
      .reduce(_+_) / labeledOutlierScores.count
    val labeledInlierScoresMean = labeledInlierScores
      .map(_.outlierScore)
      .reduce(_+_) / labeledInlierScores.count

    val uncert = 0.02
    val expectedOutlierScoreMean = 0.61
    val expectedInlierScoreMean = 0.41
    Assert.assertTrue(labeledOutlierScoresMean === expectedOutlierScoreMean +- uncert,
      s"expected labeledOutlierScoreMean = $expectedOutlierScoreMean +- $uncert, but observed" +
        s" $labeledOutlierScoresMean")
    Assert.assertTrue(labeledInlierScoresMean === expectedInlierScoreMean +- uncert,
      s"expected labeledInlierScoreMean = $expectedInlierScoreMean +/- $uncert, but observed" +
        s" $labeledInlierScoresMean")

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation of 1 from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocThreshold = 0.99
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(auroc > aurocThreshold, s"Expected area under ROC > $aurocThreshold, but" +
      s" observed $auroc")

    spark.stop()
  }
}
