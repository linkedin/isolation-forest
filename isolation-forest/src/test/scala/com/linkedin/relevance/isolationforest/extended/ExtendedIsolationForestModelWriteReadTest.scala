package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.TestUtils.{LabeledDataPointVector, ScoringResult, getSparkSession, loadMammographyData}
import com.linkedin.relevance.isolationforest.extended.{ExtendedIsolationForest, ExtendedIsolationForestModel, ExtendedIsolationTree}
import org.apache.commons.io.FileUtils.deleteDirectory
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.testng.Assert
import org.testng.annotations.Test

import java.io.File

/**
 * A suite of tests mirroring IsolationForestModelWriteReadTest but for the ExtendedIsolationForest.
 * We verify that saving and loading a trained ExtendedIsolationForestModel preserves:
 * 1) Model params (like randomSeed, contamination, etc.)
 * 2) Scores and predictions
 * 3) Tree structures
 * 4) Behavior with zero contamination and identical-feature data
 * 5) Behavior with an empty extended forest
 */
class ExtendedIsolationForestModelWriteReadTest extends Logging {

  @Test(description = "extendedIsolationForestModelWriteReadTest")
  def extendedIsolationForestModelWriteReadTest(): Unit = {

    val spark = getSparkSession
    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new extended isolation forest model
    val extendedIF = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.02)
      .setRandomSeed(1)
      .setExtensionLevel(1) // Example: partial extension

    val extendedIFModel1 = extendedIF.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/savedExtendedIsolationForestModel"
    extendedIFModel1.write.overwrite().save(savePath)
    val extendedIFModel2 = ExtendedIsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      extendedIFModel1.extractParamMap().toString,
      extendedIFModel2.extractParamMap().toString)
    Assert.assertEquals(extendedIFModel1.getNumSamples, extendedIFModel2.getNumSamples)
    Assert.assertEquals(extendedIFModel1.getNumFeatures, extendedIFModel2.getNumFeatures)
    Assert.assertEquals(
      extendedIFModel1.getOutlierScoreThreshold,
      extendedIFModel2.getOutlierScoreThreshold)

    // Calculate the AUC for both the original and saved/loaded model and assert they are equal
    val scores1 = extendedIFModel1.transform(data).as[ScoringResult]
    val metrics1 = new BinaryClassificationMetrics(scores1.rdd.map(x => (x.outlierScore, x.label)))
    val auroc1 = metrics1.areaUnderROC()

    val scores2 = extendedIFModel2.transform(data).as[ScoringResult]
    val metrics2 = new BinaryClassificationMetrics(scores2.rdd.map(x => (x.outlierScore, x.label)))
    val auroc2 = metrics2.areaUnderROC()

    Assert.assertEquals(auroc1, auroc2)

    // Assert the predicted labels are equal
    val predictedLabels1 = scores1.map(x => x.predictedLabel).collect()
    val predictedLabels2 = scores2.map(x => x.predictedLabel).collect()
    Assert.assertEquals(predictedLabels1.toSeq, predictedLabels2.toSeq)

    // Compare each tree in the original and saved/loaded model and assert they are equal
    extendedIFModel1.extendedIsolationTrees
      .zip(extendedIFModel2.extendedIsolationTrees)
      .foreach { case (tree1: ExtendedIsolationTree, tree2: ExtendedIsolationTree) =>
        Assert.assertEquals(tree2.extendedNode.toString, tree1.extendedNode.toString)
      }

    spark.stop()
  }

  @Test(description = "extendedIsolationForestModelZeroContaminationWriteReadTest")
  def extendedIsolationForestModelZeroContaminationWriteReadTest(): Unit = {

    val spark = getSparkSession
    import spark.implicits._

    val data = loadMammographyData(spark)

    // Train a new extended isolation forest model
    val extendedIF = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.0)
      .setRandomSeed(1)

    val extendedIFModel1 = extendedIF.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/savedExtendedIsolationForestModelZeroContamination"
    extendedIFModel1.write.overwrite().save(savePath)
    val extendedIFModel2 = ExtendedIsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      extendedIFModel1.extractParamMap().toString,
      extendedIFModel2.extractParamMap().toString)
    Assert.assertEquals(extendedIFModel1.getNumSamples, extendedIFModel2.getNumSamples)
    Assert.assertEquals(extendedIFModel1.getNumFeatures, extendedIFModel2.getNumFeatures)
    Assert.assertEquals(
      extendedIFModel1.getOutlierScoreThreshold,
      extendedIFModel2.getOutlierScoreThreshold)

    // Calculate the AUC for both the original and saved/loaded model and assert they are equal
    val scores1 = extendedIFModel1.transform(data).as[ScoringResult]
    val metrics1 = new BinaryClassificationMetrics(scores1.rdd.map(x => (x.outlierScore, x.label)))
    val auroc1 = metrics1.areaUnderROC()

    val scores2 = extendedIFModel2.transform(data).as[ScoringResult]
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
    extendedIFModel1.extendedIsolationTrees
      .zip(extendedIFModel2.extendedIsolationTrees)
      .foreach { case (tree1: ExtendedIsolationTree, tree2: ExtendedIsolationTree) =>
        Assert.assertEquals(tree2.extendedNode.toString, tree1.extendedNode.toString)
      }

    spark.stop()
  }

  @Test(description = "extendedIsolationForestIdenticalFeatureValuesWriteReadTest")
  def extendedIsolationForestIdenticalFeatureValuesWriteReadTest(): Unit = {

    val spark = getSparkSession
    import spark.implicits._

    val rawData = Seq(
      (0.0, 1.0, 2.0, 3.0, 1.0),
      (0.0, 1.0, 2.0, 3.0, 1.0),
      (0.0, 1.0, 2.0, 3.0, 0.0),
      (0.0, 1.0, 2.0, 3.0, 0.0),
      (0.0, 1.0, 2.0, 3.0, 1.0)).toDF("feature0", "feature1", "feature2", "feature3", "label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("feature0", "feature1", "feature2", "feature3"))
      .setOutputCol("features")

    val data = assembler
      .transform(rawData)
      .select("features", "label")
      .as[LabeledDataPointVector]

    // Train a new extended isolation forest model
    val extendedIF = new ExtendedIsolationForest()
      .setNumEstimators(100)
      .setBootstrap(false)
      .setMaxSamples(3)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
      .setContamination(0.1)
      .setRandomSeed(1)

    val extendedIFModel1 = extendedIF.fit(data)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/savedExtendedIsolationForestModelIdenticalFeatures"
    extendedIFModel1.write.overwrite().save(savePath)
    val extendedIFModel2 = ExtendedIsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // In many extended splits, if all features are identical, the node may still become a leaf.
    // So we can check something like: Are all top nodes leaves or do they remain?
    // This is optional. We'll just verify the transform output is the same, which is the key test.

    // Calculate the scores using both models and assert they are equal
    val scores1 = extendedIFModel1.transform(data).as[ScoringResult]
    val scores2 = extendedIFModel2.transform(data).as[ScoringResult]

    Assert.assertEquals(
      scores1.map(x => x.outlierScore).collect().toSeq,
      scores2.map(x => x.outlierScore).collect().toSeq)

    spark.stop()
  }

  @Test(description = "emptyExtendedIsolationForestModelWriteReadTest")
  def emptyExtendedIsolationForestModelWriteReadTest(): Unit = {

    val spark = getSparkSession

    // Create an extended isolation forest model with no isolation trees
    val extendedIFModel1 = new ExtendedIsolationForestModel("testUid", Array(), numSamples = 1, numFeatures = 2)
    extendedIFModel1.setOutlierScoreThreshold(0.5)

    // Write the trained model to disk and then read it back from disk
    val savePath = System.getProperty("java.io.tmpdir") + "/emptyExtendedIsolationForestModelWriteReadTest"
    extendedIFModel1.write.overwrite().save(savePath)
    val extendedIFModel2 = ExtendedIsolationForestModel.load(savePath)
    deleteDirectory(new File(savePath))

    // Assert that all parameter values are equal
    Assert.assertEquals(
      extendedIFModel1.extractParamMap().toString,
      extendedIFModel2.extractParamMap().toString)
    Assert.assertEquals(extendedIFModel1.getNumSamples, extendedIFModel2.getNumSamples)
    Assert.assertEquals(extendedIFModel1.getNumFeatures, extendedIFModel2.getNumFeatures)
    Assert.assertEquals(
      extendedIFModel1.getOutlierScoreThreshold,
      extendedIFModel2.getOutlierScoreThreshold)

    // Assert that the loaded model has 0 extended trees
    Assert.assertEquals(extendedIFModel2.extendedIsolationTrees.length, 0)

    spark.stop()
  }

//  @Test(description = "savedExtendedIsolationForestModelTreeStructureTest")
//  def savedExtendedIsolationForestModelTreeStructureTest(): Unit = {
//
//    val spark = getSparkSession
//
//    // Suppose you have a saved extended model in resources:
//    val modelPath = "src/test/resources/savedExtendedIsolationForestModel"
//    val extendedIFModel = ExtendedIsolationForestModel.load(modelPath)
//    val observedTreeStructure = extendedIFModel.extendedIsolationTrees.head.extendedNode.toString
//
//    // We'll read an expected structure from file (like your original test).
//    val expectedTreeStructurePath = "src/test/resources/expectedExtendedTreeStructure.txt"
//    val bufferedSource = scala.io.Source.fromFile(expectedTreeStructurePath)
//    val expectedTreeStructure = bufferedSource.mkString
//    bufferedSource.close()
//
//    Assert.assertEquals(observedTreeStructure, expectedTreeStructure)
//
//    spark.stop()
//  }
}
