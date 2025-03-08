package com.linkedin.relevance.isolationforest.integration

import com.linkedin.relevance.isolationforest.IsolationForest
import com.linkedin.relevance.isolationforest.core.TestUtils._
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.udf
import org.testng.Assert
import org.testng.annotations.Test

import java.io.File

/**
 * This is the first part of an end-to-end integration test for training an
 * Isolation Forest model on a dataset and exporting the model and data for
 * conversion to ONNX. The second part of the test is in the ONNX converter
 * module.
 */
class IsolationForestONNXIntegrationTest {

  val sparkVersion: String = System.getenv("SPARK_VERSION")
  val scalaVersionShort: String = System.getenv("SCALA_VERSION_SHORT")

  @Test(description = "exportIsolationForestModelAndDataForONNXTest")
  def exportIsolationForestModelAndData(): Unit = {
    val spark = getSparkSession
    import spark.implicits._

    // 1) Load your mammography data
    val data = loadMammographyData(spark)

    // 2) Train isolation forest
    val isolationForest = new IsolationForest()
      .setNumEstimators(100)
      .setMaxSamples(256)
      .setMaxFeatures(1.0)
      .setFeaturesCol("features")
      .setScoreCol("outlierScore")
      .setPredictionCol("predictedLabel")
      .setRandomSeed(42)
    // optional contamination, etc.

    val model = isolationForest.fit(data)

    // 3) Score
    val scoredData = model.transform(data)

    // 4) Write artifacts
    val basePath = "/tmp/isolationForestModelAndDataForONNX" + "_" + sparkVersion + "_" + scalaVersionShort
    val modelPath = s"$basePath/model"
    val csvPath   = s"$basePath/scored"

    FileUtils.deleteDirectory(new File(basePath))

    // Save the Spark model
    spark.conf.set("spark.sql.avro.compression.codec", "uncompressed")
    model.write.overwrite().save(modelPath)

    // Flatten the features to an array for indexing
    val toArray = udf { v: Vector => v.toArray }
    val flattened = scoredData
      .withColumn("feats", toArray($"features"))
      .selectExpr(
        "feats[0] as f0",
        "feats[1] as f1",
        "feats[2] as f2",
        "feats[3] as f3",
        "feats[4] as f4",
        "feats[5] as f5",
        "outlierScore as sparkScore"
      )

    flattened.write.option("header","true").mode("overwrite").csv(csvPath)

    spark.stop()

    // Possibly you assert that the model and CSV exist, or just pass
    val modelDir = new File(modelPath)
    val csvDir   = new File(csvPath)
    Assert.assertTrue(modelDir.exists(), s"Model path not found: $modelPath")
    Assert.assertTrue(csvDir.exists(),   s"Scored CSV path not found: $csvPath")
  }
}
