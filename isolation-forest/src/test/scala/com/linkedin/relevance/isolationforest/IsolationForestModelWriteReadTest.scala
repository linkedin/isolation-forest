package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Nodes.ExternalNode
import com.linkedin.relevance.isolationforest.core.TestUtils.{
  LabeledDataPointVector,
  ScoringResult,
  getSparkSession,
  loadMammographyData,
}
import org.apache.commons.io.FileUtils.deleteDirectory
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.testng.Assert
import org.testng.annotations.Test

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files

class IsolationForestModelWriteReadTest extends Logging {

  private def assertErrorContains(error: Throwable, expectedMessage: String): Unit = {
    @scala.annotation.tailrec
    def containsMessage(current: Throwable): Boolean =
      if (current == null) {
        false
      } else if (current.getMessage != null && current.getMessage.contains(expectedMessage)) {
        true
      } else {
        containsMessage(current.getCause)
      }

    Assert.assertTrue(
      containsMessage(error),
      s"Expected an exception containing '$expectedMessage', but observed: ${error}",
    )
  }

  @Test(description = "isolationForestModelWriteReadTest")
  def isolationForestModelWriteReadTest(): Unit = {

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
      .setRandomSeed(1)

    val isolationForestModel1 = isolationForest.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/savedIsolationForestModel"
    isolationForestModel1.write.overwrite().save(savePath)
    val isolationForestModel2 = IsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      isolationForestModel1.extractParamMap().toString,
      isolationForestModel2.extractParamMap().toString,
    )
    Assert.assertEquals(isolationForestModel1.getNumSamples, isolationForestModel2.getNumSamples)
    Assert.assertEquals(isolationForestModel1.getNumFeatures, isolationForestModel2.getNumFeatures)
    Assert.assertEquals(
      isolationForestModel1.getTotalNumFeatures,
      isolationForestModel2.getTotalNumFeatures,
    )
    Assert.assertEquals(
      isolationForestModel1.getOutlierScoreThreshold,
      isolationForestModel2.getOutlierScoreThreshold,
    )

    // Calculate the AUC for both the original and saved/loaded model and assert they are equal
    val scores1 = isolationForestModel1.transform(data).as[ScoringResult]
    val metrics1 = new BinaryClassificationMetrics(scores1.rdd.map(x => (x.outlierScore, x.label)))
    val auroc1 = metrics1.areaUnderROC()

    val scores2 = isolationForestModel2.transform(data).as[ScoringResult]
    val metrics2 = new BinaryClassificationMetrics(scores2.rdd.map(x => (x.outlierScore, x.label)))
    val auroc2 = metrics2.areaUnderROC()

    Assert.assertEquals(auroc1, auroc2)

    // Assert the predicted labels are equal
    val predictedLabels1 = scores1.map(x => x.predictedLabel).collect()
    val predictedLabels2 = scores2.map(x => x.predictedLabel).collect()
    Assert.assertEquals(predictedLabels1.toSeq, predictedLabels2.toSeq)

    // Compare each tree in the original and saved/loaded model and assert they are equal
    isolationForestModel1.isolationTrees
      .zip(isolationForestModel2.isolationTrees)
      .foreach { case (tree1: IsolationTree, tree2: IsolationTree) =>
        Assert.assertEquals(tree2.node.toString, tree1.node.toString)
      }

    spark.stop()
  }

  @Test(description = "isolationForestModelZeroContaminationWriteReadTest")
  def isolationForestModelZeroContaminationWriteReadTest(): Unit = {

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

    val isolationForestModel1 = isolationForest.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath =
      System.getProperty("java.io.tmpdir") + "/savedIsolationForestModelZeroContamination"
    isolationForestModel1.write.overwrite().save(savePath)
    val isolationForestModel2 = IsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      isolationForestModel1.extractParamMap().toString,
      isolationForestModel2.extractParamMap().toString,
    )
    Assert.assertEquals(isolationForestModel1.getNumSamples, isolationForestModel2.getNumSamples)
    Assert.assertEquals(isolationForestModel1.getNumFeatures, isolationForestModel2.getNumFeatures)
    Assert.assertEquals(
      isolationForestModel1.getTotalNumFeatures,
      isolationForestModel2.getTotalNumFeatures,
    )
    Assert.assertEquals(
      isolationForestModel1.getOutlierScoreThreshold,
      isolationForestModel2.getOutlierScoreThreshold,
    )

    // Calculate the AUC for both the original and saved/loaded model and assert they are equal
    val scores1 = isolationForestModel1.transform(data).as[ScoringResult]
    val metrics1 = new BinaryClassificationMetrics(scores1.rdd.map(x => (x.outlierScore, x.label)))
    val auroc1 = metrics1.areaUnderROC()

    val scores2 = isolationForestModel2.transform(data).as[ScoringResult]
    val metrics2 = new BinaryClassificationMetrics(scores2.rdd.map(x => (x.outlierScore, x.label)))
    val auroc2 = metrics2.areaUnderROC()

    Assert.assertEquals(auroc1, auroc2)

    // Assert the predicted labels are equal and always 0.0
    val predictedLabels1 = scores1.map(x => x.predictedLabel).collect()
    val predictedLabels2 = scores2.map(x => x.predictedLabel).collect()
    val expectedLabels = Array.fill[Double](predictedLabels1.length)(0.0)
    Assert.assertEquals(predictedLabels1.toSeq, predictedLabels2.toSeq)
    Assert.assertEquals(predictedLabels2.toSeq, expectedLabels.toSeq)

    // Compare each tree in the original and saved/loaded model and assert they are equal
    isolationForestModel1.isolationTrees
      .zip(isolationForestModel2.isolationTrees)
      .foreach { case (tree1: IsolationTree, tree2: IsolationTree) =>
        Assert.assertEquals(tree2.node.toString, tree1.node.toString)
      }

    spark.stop()
  }

  @Test(description = "isolationForestIdenticalFeatureValuesWriteReadTest")
  def isolationForestIdenticalFeatureValuesWriteReadTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val rawData = Seq(
      (0.0, 1.0, 2.0, 3.0, 1.0),
      (0.0, 1.0, 2.0, 3.0, 1.0),
      (0.0, 1.0, 2.0, 3.0, 0.0),
      (0.0, 1.0, 2.0, 3.0, 0.0),
      (0.0, 1.0, 2.0, 3.0, 1.0),
    ).toDF("feature0", "feature1", "feature2", "feature3", "label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("feature0", "feature1", "feature2", "feature3"))
      .setOutputCol("features")

    val data = assembler
      .transform(rawData)
      .select("features", "label")
      .as[LabeledDataPointVector]

    // Train a new isolation forest model
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(3)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.1)
      .setRandomSeed(1)

    val isolationForestModel1 = isolationForest.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath =
      System.getProperty("java.io.tmpdir") + "/savedIsolationForestModelIdenticalFeatures"
    isolationForestModel1.write.overwrite().save(savePath)
    val isolationForestModel2 = IsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that the root node of every tree is an external node (because no splits could be made)
    isolationForestModel1.isolationTrees.foreach(x =>
      Assert.assertTrue(x.node.isInstanceOf[ExternalNode]),
    )
    isolationForestModel2.isolationTrees.foreach(x =>
      Assert.assertTrue(x.node.isInstanceOf[ExternalNode]),
    )

    // Calculate the scores using both models and assert they are equal
    val scores1 = isolationForestModel1.transform(data).as[ScoringResult]
    val scores2 = isolationForestModel2.transform(data).as[ScoringResult]

    Assert.assertEquals(
      scores1.map(x => x.outlierScore).collect().toSeq,
      scores2.map(x => x.outlierScore).collect().toSeq,
    )

    spark.stop()
  }

  @Test(description = "emptyIsolationForestModelWriteReadTest")
  def emptyIsolationForestModelWriteReadTest(): Unit = {

    val spark = getSparkSession

    // Create an isolation forest model with no isolation trees
    val isolationForestModel1 =
      new IsolationForestModel(
        "testUid",
        Array(),
        numSamples = 2,
        numFeatures = 2,
        totalNumFeatures = 2,
      )
    isolationForestModel1.setOutlierScoreThreshold(0.5)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/emptyIsolationForestModelWriteReadTest"
    isolationForestModel1.write.overwrite().save(savePath)
    val isolationForestModel2 = IsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      isolationForestModel1.extractParamMap().toString,
      isolationForestModel2.extractParamMap().toString,
    )
    Assert.assertEquals(isolationForestModel1.getNumSamples, isolationForestModel2.getNumSamples)
    Assert.assertEquals(isolationForestModel1.getNumFeatures, isolationForestModel2.getNumFeatures)
    Assert.assertEquals(
      isolationForestModel1.getTotalNumFeatures,
      isolationForestModel2.getTotalNumFeatures,
    )
    Assert.assertEquals(
      isolationForestModel1.getOutlierScoreThreshold,
      isolationForestModel2.getOutlierScoreThreshold,
    )

    // Assert that the loaded model has 0 trees
    Assert.assertEquals(isolationForestModel2.isolationTrees.length, 0)

    spark.stop()
  }

  @Test(
    description = "emptyIsolationForestModelTransformThrowsTest",
    expectedExceptions = Array(classOf[IllegalArgumentException]),
  )
  def emptyIsolationForestModelTransformThrowsTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = Seq(Tuple1(Vectors.dense(1.0, 2.0))).toDF("features")
    val emptyModel =
      new IsolationForestModel(
        "testUid",
        Array(),
        numSamples = 2,
        numFeatures = 2,
        totalNumFeatures = 2,
      )

    try
      emptyModel.transform(data)
    finally
      spark.stop()
  }

  @Test(
    description = "isolationForestModelNumSamplesOneTransformThrowsTest",
    expectedExceptions = Array(classOf[IllegalArgumentException]),
  )
  def isolationForestModelNumSamplesOneTransformThrowsTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val data = Seq(Tuple1(Vectors.dense(1.0, 2.0))).toDF("features")
    val invalidModel = new IsolationForestModel(
      "testUid",
      Array(new IsolationTree(ExternalNode(2))),
      numSamples = 1,
      numFeatures = 2,
      totalNumFeatures = 2,
    )

    try
      invalidModel.transform(data)
    finally
      spark.stop()
  }

  @Test(description = "isolationForestModelFeatureDimensionValidationTest")
  def isolationForestModelFeatureDimensionValidationTest(): Unit = {

    val spark = getSparkSession

    import spark.implicits._

    val validModel = new IsolationForestModel(
      "testUid",
      Array(new IsolationTree(ExternalNode(2))),
      numSamples = 2,
      numFeatures = 2,
      totalNumFeatures = 2,
    )

    try
      Seq(
        Vectors.dense(1.0),
        Vectors.dense(1.0, 2.0, 3.0),
      ).foreach { invalidVector =>
        try {
          validModel.transform(Seq(Tuple1(invalidVector)).toDF("features")).collect()
          Assert.fail(s"Expected feature-dimension validation to fail for $invalidVector.")
        } catch {
          case error: Exception =>
            assertErrorContains(error, "did not match the model's training dimension 2")
        }
      }
    finally
      spark.stop()
  }

  @Test(description = "legacyIsolationForestModelConstructorUsesUnknownTotalNumFeaturesTest")
  def legacyIsolationForestModelConstructorUsesUnknownTotalNumFeaturesTest(): Unit = {

    val legacyModel =
      new IsolationForestModel("testUid", Array(new IsolationTree(ExternalNode(2))), 2, 2)

    Assert.assertFalse(legacyModel.hasKnownTotalNumFeatures)
    Assert.assertEquals(
      legacyModel.getTotalNumFeatures,
      IsolationForestModel.UnknownTotalNumFeatures,
    )
  }

  @Test(description = "savedIsolationForestModelTreeStructureTest")
  def savedIsolationForestModelTreeStructureTest(): Unit = {

    val spark = getSparkSession

    val modelPath = "src/test/resources/savedIsolationForestModel"
    val isolationForestModel = IsolationForestModel.load(modelPath)
    val observedTreeStructure = isolationForestModel.isolationTrees(0).node.toString

    val expectedTreeStructurePath = "src/test/resources/expectedTreeStructure.txt"
    val bufferedSource = scala.io.Source.fromFile(expectedTreeStructurePath)
    val expectedTreeStructure = bufferedSource.mkString
    bufferedSource.close()

    Assert.assertEquals(observedTreeStructure, expectedTreeStructure)

    spark.stop()
  }

  @Test(description = "legacyIsolationForestModelWithoutTotalNumFeaturesLoadsTest")
  def legacyIsolationForestModelWithoutTotalNumFeaturesLoadsTest(): Unit = {

    val spark = getSparkSession
    val data = loadMammographyData(spark)
    val savePath =
      System.getProperty("java.io.tmpdir") + "/savedIsolationForestModelLegacyMetadata"
    val legacyModelDir = new File(savePath)

    try {
      val isolationForest = new IsolationForest()
        .setNumEstimators(20)
        .setBootstrap(false)
        .setMaxSamples(256)
        .setMaxFeatures(0.5)
        .setFeaturesCol("features")
        .setPredictionCol("predictedLabel")
        .setScoreCol("outlierScore")
        .setContamination(0.02)
        .setRandomSeed(1)

      val isolationForestModel = isolationForest.fit(data)
      Assert.assertTrue(
        isolationForestModel.getNumFeatures < isolationForestModel.getTotalNumFeatures,
      )
      isolationForestModel.write.overwrite().save(savePath)

      val metadataPath = new File(legacyModelDir, "metadata/part-00000").toPath
      val currentMetadata = new String(Files.readAllBytes(metadataPath), StandardCharsets.UTF_8)
      val legacyMetadata = currentMetadata.replaceFirst(",\"totalNumFeatures\":\\d+", "")
      Files.write(metadataPath, legacyMetadata.getBytes(StandardCharsets.UTF_8))
      new File(legacyModelDir, "metadata").listFiles().foreach { file =>
        if (file.getName.startsWith(".")) {
          file.delete()
        }
      }

      val legacyIsolationForestModel = IsolationForestModel.load(legacyModelDir.getPath)
      Assert.assertFalse(legacyIsolationForestModel.hasKnownTotalNumFeatures)
      Assert.assertEquals(
        legacyIsolationForestModel.getTotalNumFeatures,
        IsolationForestModel.UnknownTotalNumFeatures,
      )
      Assert.assertEquals(legacyIsolationForestModel.transform(data).count(), data.count())
    } finally {
      if (legacyModelDir.exists()) {
        deleteDirectory(legacyModelDir)
      }
      spark.stop()
    }
  }
}
