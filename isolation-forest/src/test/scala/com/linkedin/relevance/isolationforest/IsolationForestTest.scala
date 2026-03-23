package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.TestUtils._
import org.apache.commons.io.FileUtils.deleteDirectory
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._
import org.testng.Assert
import org.testng.annotations.Test

import java.io.File

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

    isolationForest1.write.overwrite().save(savePath)
    val isolationForest2 = IsolationForest.load(savePath)
    deleteDirectory(new File(savePath))

    Assert.assertEquals(
      isolationForest1.extractParamMap().toString,
      isolationForest2.extractParamMap().toString,
    )

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
    Assert.assertTrue(
      auroc === aurocExpectation +- uncert,
      "expected area under ROC =" +
        s" $aurocExpectation +/- $uncert, but observed $auroc",
    )

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
    Assert.assertTrue(
      auroc === aurocExpectation +- uncert,
      "expected area under ROC =" +
        s" $aurocExpectation +/- $uncert, but observed $auroc",
    )

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
    val predictedLabels = scores.map(x => x.predictedLabel).collect()
    val expectedLabels = Array.fill[Double](predictedLabels.length)(0.0)

    Assert.assertEquals(
      predictedLabels.toSeq,
      expectedLabels.toSeq,
      "expected all predicted labels to be 0.0",
    )

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
      .reduce(_ + _) / labeledOutlierScores.count()
    val labeledInlierScoresMean = labeledInlierScores
      .map(_.outlierScore)
      .reduce(_ + _) / labeledInlierScores.count()

    val uncert = 0.02
    val expectedOutlierScoreMean = 0.61
    val expectedInlierScoreMean = 0.41
    Assert.assertTrue(
      labeledOutlierScoresMean === expectedOutlierScoreMean +- uncert,
      s"expected labeledOutlierScoreMean = $expectedOutlierScoreMean +- $uncert, but observed" +
        s" $labeledOutlierScoresMean",
    )
    Assert.assertTrue(
      labeledInlierScoresMean === expectedInlierScoreMean +- uncert,
      s"expected labeledInlierScoreMean = $expectedInlierScoreMean +/- $uncert, but observed" +
        s" $labeledInlierScoresMean",
    )

    // Calculate area under ROC curve and assert
    val scores = isolationForestModel.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

    // Expectation of 1 from results in the 2008 "Isolation Forest" paper by F. T. Liu, et al.
    val aurocThreshold = 0.99
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(
      auroc > aurocThreshold,
      s"Expected area under ROC > $aurocThreshold, but" +
        s" observed $auroc",
    )

    spark.stop()
  }

  @Test(
    description = "isolationForestNumSamplesOneThrowsTest",
    expectedExceptions = Array(classOf[IllegalArgumentException]),
  )
  def isolationForestNumSamplesOneThrowsTest(): Unit = {

    val spark = getSparkSession

    val data = loadMammographyData(spark)

    val isolationForest = new IsolationForest()
      .setNumEstimators(10)
      .setBootstrap(false)
      .setMaxSamples(1.5)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setRandomSeed(1)

    try
      isolationForest.fit(data)
    finally
      spark.stop()
  }

  @Test(description = "isolationForestSeedReproducibilityTest")
  def isolationForestSeedReproducibilityTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    val modelA = new IsolationForest()
      .setNumEstimators(50)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setRandomSeed(42)
      .fit(data)

    val modelB = new IsolationForest()
      .setNumEstimators(50)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setRandomSeed(42)
      .fit(data)

    val scoresA = modelA.transform(data).as[ScoringResult].map(_.outlierScore).collect()
    val scoresB = modelB.transform(data).as[ScoringResult].map(_.outlierScore).collect()

    val maxDiff = scoresA.zip(scoresB).map { case (a, b) => math.abs(a - b) }.max
    Assert.assertTrue(
      maxDiff < 1e-10,
      s"Same seed should produce identical scores, but max diff = $maxDiff",
    )

    spark.stop()
  }

  @Test(description = "isolationForestScoreRangeTest")
  def isolationForestScoreRangeTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    val model = new IsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult].map(_.outlierScore).collect()

    Assert.assertTrue(
      scores.forall(s => s >= 0.0 && s <= 1.0),
      s"All scores should be in [0, 1], but found min=${scores.min}, max=${scores.max}",
    )

    spark.stop()
  }

  @Test(description = "isolationForestLowDimensional1DTest")
  def isolationForestLowDimensional1DTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val normal = (1 to 300).map(_ => Vectors.dense(scala.util.Random.nextGaussian()))
    val outliers = (1 to 30).map(_ => Vectors.dense(scala.util.Random.nextGaussian() + 5.0))
    val allPoints = normal ++ outliers
    val labels = Seq.fill(300)(0.0) ++ Seq.fill(30)(1.0)
    val data = allPoints.zip(labels).toDF("features", "label")

    val model = new IsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(
      auroc > 0.7,
      s"Expected AUROC > 0.7 for 1D data with separated outliers, but observed $auroc",
    )

    spark.stop()
  }

  @Test(description = "isolationForestConstantFeaturesTest")
  def isolationForestConstantFeaturesTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    // 5D data where features 3 and 4 are constant
    val rng = new java.util.Random(42)
    val allPoints = (1 to 300).map { i =>
      val base =
        if (i <= 280) Array.fill(5)(rng.nextGaussian())
        else Array.fill(5)(rng.nextGaussian() + 4.0)
      base(3) = 0.0
      base(4) = 1.0
      Vectors.dense(base)
    }
    val labels = Seq.fill(280)(0.0) ++ Seq.fill(20)(1.0)
    val data = allPoints.zip(labels).toDF("features", "label")

    val model = new IsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(
      auroc > 0.7,
      s"Expected AUROC > 0.7 with constant features, but observed $auroc",
    )

    spark.stop()
  }

  @Test(description = "isolationForestAllConstantFeaturesTest")
  def isolationForestAllConstantFeaturesTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = (1 to 300).map(_ => (Vectors.dense(1.0, 2.0, 3.0), 0.0)).toDF("features", "label")

    val model = new IsolationForest()
      .setNumEstimators(10)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult].map(_.outlierScore).collect()

    // All-constant data should not crash; scores should all be the same
    val scoreStd = {
      val mean = scores.sum / scores.length
      math.sqrt(scores.map(x => math.pow(x - mean, 2)).sum / scores.length)
    }
    Assert.assertTrue(
      scoreStd < 0.01,
      s"Expected near-zero score variance for all-constant data, but std = $scoreStd",
    )

    spark.stop()
  }

  @Test(description = "isolationForestTinyDatasetTest")
  def isolationForestTinyDatasetTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = Seq(
      (Vectors.dense(1.0, 2.0), 0.0),
      (Vectors.dense(10.0, 20.0), 1.0),
      (Vectors.dense(1.5, 2.5), 0.0),
    ).toDF("features", "label")

    val model = new IsolationForest()
      .setNumEstimators(10)
      .setMaxSamples(2)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult].map(_.outlierScore).collect()
    Assert.assertEquals(scores.length, 3, "Should score all 3 rows")
    Assert.assertTrue(
      scores.forall(s => s >= 0.0 && s <= 1.0),
      s"Scores should be in [0, 1] for tiny dataset",
    )

    spark.stop()
  }
}
