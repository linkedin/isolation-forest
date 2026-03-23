package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.TestUtils._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
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
    Assert.assertTrue(
      auroc === aurocExpectation +- uncert,
      "expected area under ROC =" +
        s" $aurocExpectation +/- $uncert, but observed $auroc",
    )

    spark.stop()
  }

  @Test(description = "extendedIsolationForestMammographyZeroExtensionDataTest")
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
    Assert.assertTrue(
      auroc === aurocExpectation +- uncert,
      "expected area under ROC =" +
        s" $aurocExpectation +/- $uncert, but observed $auroc",
    )

    spark.stop()
  }

  @Test(description = "extendedIsolationForestMammographyExactContaminationDataTest")
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
    Assert.assertTrue(
      auroc === aurocExpectation +- uncert,
      "expected area under ROC =" +
        s" $aurocExpectation +/- $uncert, but observed $auroc",
    )

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
      "expected all predicted labels to be 0.0",
    )

    spark.stop()
  }

  @Test(
    description = "extendedIsolationForestExtensionLevelTooLargeThrowsTest",
    expectedExceptions = Array(classOf[IllegalArgumentException]),
  )
  def extendedIsolationForestExtensionLevelTooLargeThrowsTest(): Unit = {

    val spark = getSparkSession

    val data = loadMammographyData(spark)

    // Mammography has 6 features, so max extensionLevel is 5. Setting 6 should throw.
    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(10)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setExtensionLevel(6)
      .setRandomSeed(1)

    try
      extendedIsolationForest.fit(data)
    finally
      spark.stop()
  }

  @Test(description = "extendedIsolationForestIntermediateExtensionLevelTest")
  def extendedIsolationForestIntermediateExtensionLevelTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    // Mammography has 6 features. Test intermediate extensionLevel values (1 through 4)
    // to verify they produce valid models with reasonable AUROC.
    for (extLevel <- 1 to 4) {
      val extendedIsolationForest = new ExtendedIsolationForest()
        .setNumEstimators(100)
        .setBootstrap(false)
        .setMaxSamples(256)
        .setMaxFeatures(1.0)
        .setFeaturesCol("features")
        .setPredictionCol("predictedLabel")
        .setScoreCol("outlierScore")
        .setContamination(0.0)
        .setExtensionLevel(extLevel)
        .setRandomSeed(1)

      val model = extendedIsolationForest.fit(data)

      // Verify extensionLevel was persisted correctly on the trained model
      Assert.assertEquals(
        model.getExtensionLevel,
        extLevel,
        s"extensionLevel should be $extLevel on the trained model",
      )

      val scores = model.transform(data).as[ScoringResult]
      val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))

      // All intermediate levels should produce a reasonable model (AUROC > 0.7)
      val auroc = metrics.areaUnderROC()
      Assert.assertTrue(
        auroc > 0.7,
        s"Expected AUROC > 0.7 for extensionLevel=$extLevel, but observed $auroc",
      )
    }

    spark.stop()
  }

  @Test(description = "extendedIsolationForestDefaultExtensionLevelDoesNotLeakAcrossFitsTest")
  def extendedIsolationForestDefaultExtensionLevelDoesNotLeakAcrossFitsTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val fourFeatureData = Seq(
      (0.0, 1.0, 2.0, 3.0),
      (1.0, 2.0, 3.0, 4.0),
      (2.0, 3.0, 4.0, 5.0),
      (3.0, 4.0, 5.0, 6.0),
      (4.0, 5.0, 6.0, 7.0),
    ).toDF("f0", "f1", "f2", "f3")
    val twoFeatureData = Seq(
      (0.0, 1.0),
      (1.0, 2.0),
      (2.0, 3.0),
      (3.0, 4.0),
      (4.0, 5.0),
    ).toDF("f0", "f1")

    val fourFeatureAssembler = new VectorAssembler()
      .setInputCols(Array("f0", "f1", "f2", "f3"))
      .setOutputCol("features")
    val twoFeatureAssembler = new VectorAssembler()
      .setInputCols(Array("f0", "f1"))
      .setOutputCol("features")

    val fourFeatureDataset = fourFeatureAssembler.transform(fourFeatureData).select("features")
    val twoFeatureDataset = twoFeatureAssembler.transform(twoFeatureData).select("features")

    val extendedIsolationForest = new ExtendedIsolationForest()
      .setNumEstimators(5)
      .setBootstrap(false)
      .setMaxSamples(4)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setRandomSeed(1)

    Assert.assertFalse(
      extendedIsolationForest.isSet(extendedIsolationForest.extensionLevel),
      "extensionLevel should not be set before fitting when the user did not specify it",
    )

    val fourFeatureModel = extendedIsolationForest.fit(fourFeatureDataset)
    Assert.assertEquals(
      fourFeatureModel.getExtensionLevel,
      3,
      "default extensionLevel should resolve to numFeatures - 1 for the first fit",
    )
    Assert.assertFalse(
      extendedIsolationForest.isSet(extendedIsolationForest.extensionLevel),
      "fit() should not mutate the estimator with a resolved default extensionLevel",
    )

    val twoFeatureModel = extendedIsolationForest.fit(twoFeatureDataset)
    Assert.assertEquals(
      twoFeatureModel.getExtensionLevel,
      1,
      "default extensionLevel should be re-resolved for each fit based on that dataset",
    )
    Assert.assertFalse(
      extendedIsolationForest.isSet(extendedIsolationForest.extensionLevel),
      "estimator should remain unset after repeated fits when extensionLevel was not user-specified",
    )

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
    Assert.assertTrue(
      auroc > aurocThreshold,
      s"Expected area under ROC > $aurocThreshold, but" +
        s" observed $auroc",
    )

    spark.stop()
  }

  @Test(
    description = "extendedIsolationForestNumSamplesOneThrowsTest",
    expectedExceptions = Array(classOf[IllegalArgumentException]),
  )
  def extendedIsolationForestNumSamplesOneThrowsTest(): Unit = {

    val spark = getSparkSession

    val data = loadMammographyData(spark)

    val extendedIsolationForest = new ExtendedIsolationForest()
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
      extendedIsolationForest.fit(data)
    finally
      spark.stop()
  }

  @Test(description = "extendedIsolationForestSeedReproducibilityTest")
  def extendedIsolationForestSeedReproducibilityTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    val modelA = new ExtendedIsolationForest()
      .setNumEstimators(50)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setExtensionLevel(5)
      .setRandomSeed(42)
      .fit(data)

    val modelB = new ExtendedIsolationForest()
      .setNumEstimators(50)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setExtensionLevel(5)
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

  @Test(description = "extendedIsolationForestScoreRangeTest")
  def extendedIsolationForestScoreRangeTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = loadMammographyData(spark)

    val model = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setExtensionLevel(5)
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult].map(_.outlierScore).collect()

    Assert.assertTrue(
      scores.forall(s => s >= 0.0 && s <= 1.0),
      s"All scores should be in [0, 1], but found min=${scores.min}, max=${scores.max}",
    )

    spark.stop()
  }

  @Test(description = "extendedIsolationForestLowDimensional1DTest")
  def extendedIsolationForestLowDimensional1DTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val normal = (1 to 300).map(_ => Vectors.dense(scala.util.Random.nextGaussian()))
    val outliers = (1 to 30).map(_ => Vectors.dense(scala.util.Random.nextGaussian() + 5.0))
    val allPoints = normal ++ outliers
    val labels = Seq.fill(300)(0.0) ++ Seq.fill(30)(1.0)
    val data = allPoints.zip(labels).toDF("features", "label")

    // 1D data: only valid extensionLevel is 0
    val model = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setExtensionLevel(0)
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

  @Test(description = "extendedIsolationForestLowDimensional2DTest")
  def extendedIsolationForestLowDimensional2DTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val rng = new java.util.Random(42)
    val normal = (1 to 300).map(_ => Vectors.dense(rng.nextGaussian(), rng.nextGaussian()))
    val outliers =
      (1 to 30).map(_ => Vectors.dense(rng.nextGaussian() + 5.0, rng.nextGaussian() + 5.0))
    val allPoints = normal ++ outliers
    val labels = Seq.fill(300)(0.0) ++ Seq.fill(30)(1.0)
    val data = allPoints.zip(labels).toDF("features", "label")

    // 2D data: extensionLevel=1 is fully extended
    val model = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setExtensionLevel(1)
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult]
    val metrics = new BinaryClassificationMetrics(scores.rdd.map(x => (x.outlierScore, x.label)))
    val auroc = metrics.areaUnderROC()
    Assert.assertTrue(
      auroc > 0.7,
      s"Expected AUROC > 0.7 for 2D data with separated outliers, but observed $auroc",
    )

    spark.stop()
  }

  @Test(description = "extendedIsolationForestConstantFeaturesTest")
  def extendedIsolationForestConstantFeaturesTest(): Unit = {

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

    val model = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setExtensionLevel(4)
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

  @Test(description = "extendedIsolationForestAllConstantFeaturesTest")
  def extendedIsolationForestAllConstantFeaturesTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = (1 to 300).map(_ => (Vectors.dense(1.0, 2.0, 3.0), 0.0)).toDF("features", "label")

    val model = new ExtendedIsolationForest()
      .setNumEstimators(10)
      .setMaxSamples(256)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setExtensionLevel(2)
      .setRandomSeed(1)
      .fit(data)

    val scores = model.transform(data).as[ScoringResult].map(_.outlierScore).collect()

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

  @Test(description = "extendedIsolationForestTinyDatasetTest")
  def extendedIsolationForestTinyDatasetTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = Seq(
      (Vectors.dense(1.0, 2.0), 0.0),
      (Vectors.dense(10.0, 20.0), 1.0),
      (Vectors.dense(1.5, 2.5), 0.0),
    ).toDF("features", "label")

    val model = new ExtendedIsolationForest()
      .setNumEstimators(10)
      .setMaxSamples(2)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setExtensionLevel(0)
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
