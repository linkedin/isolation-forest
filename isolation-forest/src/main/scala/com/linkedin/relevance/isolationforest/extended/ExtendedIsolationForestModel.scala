package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.Utils.{
  DataPoint,
  avgPathLength,
  validateAndTransformSchema,
}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A trained extended isolation forest model, composed of multiple ExtendedIsolationTree instances
 * that each use random hyperplane splits.
 *
 * @param uid
 *   The immutable unique ID for the model.
 * @param extendedIsolationTrees
 *   The array of extended isolation tree models that compose the extended isolation forest.
 * @param numSamples
 *   The number of samples used to train each tree.
 * @param numFeatures
 *   The number of features used in each tree's hyperplane subspace.
 */
class ExtendedIsolationForestModel(
  override val uid: String,
  val extendedIsolationTrees: Array[ExtendedIsolationTree],
  private val numSamples: Int,
  private val numFeatures: Int,
) extends Model[ExtendedIsolationForestModel]
    with ExtendedIsolationForestParams
    with MLWritable {

  require(numSamples > 0, s"parameter numSamples must be >0, but given invalid value ${numSamples}")
  final def getNumSamples: Int = numSamples

  require(
    numFeatures > 0,
    s"parameter numFeatures must be >0, but given invalid value ${numFeatures}",
  )
  final def getNumFeatures: Int = numFeatures

  // The outlierScoreThreshold needs to be a mutable variable because it is not known when an
  // ExtendedIsolationForestModel instance is created.
  private var outlierScoreThreshold: Double = -1
  private[isolationforest] def setOutlierScoreThreshold(value: Double): Unit = {

    require(
      value == -1 || (value >= 0 && value <= 1),
      "parameter outlierScoreThreshold must be" +
        " equal to -1 (no threshold) or be in the range [0, 1]," +
        s" but given invalid value ${value}",
    )
    outlierScoreThreshold = value
  }
  final def getOutlierScoreThreshold: Double = outlierScoreThreshold

  override def copy(extra: ParamMap): ExtendedIsolationForestModel = {
    val extendedIsolationForestCopy = new ExtendedIsolationForestModel(
      uid,
      extendedIsolationTrees,
      numSamples,
      numFeatures,
    ).setParent(this.parent)
    extendedIsolationForestCopy.setOutlierScoreThreshold(outlierScoreThreshold)

    copyValues(extendedIsolationForestCopy, extra)
  }

  /**
   * Scores new data instances using the trained extended isolation forest model.
   *
   * @param data
   *   The input DataFrame of data to be scored. It must have a column $(featuresCol) that contains
   *   a feature vector for each data instance.
   * @return
   *   The same DataFrame with $(predictionCol) and $(scoreCol) appended.
   */
  override def transform(data: Dataset[_]): DataFrame = {

    transformSchema(data.schema, logging = true)

    val avgPath = avgPathLength(numSamples)
    val broadcastExtendedIsolationTrees = data.sparkSession.sparkContext
      .broadcast(extendedIsolationTrees)

    val calculatePathLength = (features: Vector) => {
      val pathLength = broadcastExtendedIsolationTrees.value
        .map(y => y.calculatePathLength(DataPoint(features.toArray.map(x => x.toFloat))))
        .sum / broadcastExtendedIsolationTrees.value.length
      Math.pow(2, -pathLength / avgPath)
    }
    val transformUDF = udf(calculatePathLength)

    val dataWithScore = data.withColumn($(scoreCol), transformUDF(col($(featuresCol))))
    val dataWithScoresAndPrediction = if (outlierScoreThreshold > 0) {
      dataWithScore.withColumn(
        $(predictionCol),
        (col($(scoreCol)) >= outlierScoreThreshold).cast("double"),
      )
    } else {
      // If threshold is not set, default predictedLabel to 0.0
      dataWithScore.withColumn($(predictionCol), lit(0.0))
    }

    dataWithScoresAndPrediction
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, $(featuresCol), $(predictionCol), $(scoreCol))

  /**
   * Returns an IsolationForestModelWriter instance that can be used to write the isolation forest
   * to disk.
   *
   * @return
   *   An IsolationForestModelWriter instance.
   */
  override def write: MLWriter =
    new ExtendedIsolationForestModelReadWrite.ExtendedIsolationForestModelWriter(this)
}

/**
 * Companion object to the ExtendedIsolationForestModel class.
 */
case object ExtendedIsolationForestModel extends MLReadable[ExtendedIsolationForestModel] {

  /**
   * Returns an ExtendedIsolationForestModelReader instance that can be used to read a saved
   * extended isolation forest from disk.
   *
   * @return
   *   An ExtendedIsolationForestModelReader instance.
   */
  override def read: MLReader[ExtendedIsolationForestModel] =
    new ExtendedIsolationForestModelReadWrite.ExtendedIsolationForestModelReader

  /**
   * Loads a saved extended isolation forest model from disk. A shortcut of `read.load(path)`.
   *
   * @param path
   *   The path to the saved extended isolation forest model.
   * @return
   *   The loaded ExtendedIsolationForestModel instance.
   */
  override def load(path: String): ExtendedIsolationForestModel = super.load(path)
}
