package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.Utils.{
  DataPoint,
  avgPathLength,
  validateFeatureVectorSize,
  validateAndTransformSchema,
}
import com.linkedin.relevance.isolationforest.core.IsolationForestParamsBase
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A trained isolation tree model. It extends the spark.ml Model class.
 *
 * @param uid
 *   The immutable unique ID for the model.
 * @param isolationTrees
 *   The array of isolation tree models that compose the isolation forest.
 * @param numSamples
 *   The number of samples used to train each tree.
 * @param numFeatures
 *   The user-specified number of features used to train each isolation tree. For certain edge
 *   cases, a given isolation tree may not have any nodes using some of these features, e.g., a
 *   shallow tree where the number of features in the training data exceeds the number of nodes in
 *   the tree.
 * @param totalNumFeatures
 *   The total number of input features seen during training, or
 *   [[IsolationForestModel.UnknownTotalNumFeatures]] for legacy loaded models that predate this
 *   metadata.
 */
class IsolationForestModel private[isolationforest] (
  override val uid: String,
  val isolationTrees: Array[IsolationTree],
  private val numSamples: Int,
  private val numFeatures: Int,
  private val totalNumFeatures: Int,
) extends Model[IsolationForestModel]
    with IsolationForestParamsBase
    with MLWritable {

  def this(
    uid: String,
    isolationTrees: Array[IsolationTree],
    numSamples: Int,
    numFeatures: Int,
  ) =
    this(
      uid,
      isolationTrees,
      numSamples,
      numFeatures,
      IsolationForestModel.UnknownTotalNumFeatures,
    )

  require(numSamples > 0, s"parameter numSamples must be >0, but given invalid value ${numSamples}")
  final def getNumSamples: Int = numSamples

  require(
    numFeatures > 0,
    s"parameter numFeatures must be >0, but given invalid value ${numFeatures}",
  )
  final def getNumFeatures: Int = numFeatures

  require(
    totalNumFeatures == IsolationForestModel.UnknownTotalNumFeatures || totalNumFeatures > 0,
    s"parameter totalNumFeatures must be >0 or UnknownTotalNumFeatures, but given invalid value ${totalNumFeatures}",
  )
  require(
    totalNumFeatures == IsolationForestModel.UnknownTotalNumFeatures || numFeatures <= totalNumFeatures,
    s"parameter numFeatures must be <= totalNumFeatures, but given invalid values" +
      s" numFeatures=${numFeatures}, totalNumFeatures=${totalNumFeatures}",
  )
  final def getTotalNumFeatures: Int = totalNumFeatures
  final def hasKnownTotalNumFeatures: Boolean =
    totalNumFeatures != IsolationForestModel.UnknownTotalNumFeatures

  // The outlierScoreThreshold needs to be a mutable variable because it is not known when an
  // IsolationForestModel instance is created.
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

  override def copy(extra: ParamMap): IsolationForestModel = {

    val isolationForestCopy =
      new IsolationForestModel(uid, isolationTrees, numSamples, numFeatures, totalNumFeatures)
        .setParent(this.parent)
    isolationForestCopy.setOutlierScoreThreshold(outlierScoreThreshold)
    copyValues(isolationForestCopy, extra)
  }

  /**
   * Scores new data instances using the trained isolation forest model.
   *
   * @param data
   *   The input DataFrame of data to be scored. It must have a column $(featuresCol) that contains
   *   a the feature vector for each data instance.
   * @return
   *   The same DataFrame with $(predictionCol) and $(scoreCol) appended.
   */
  override def transform(data: Dataset[_]): DataFrame = {

    require(
      numSamples >= 2,
      s"Cannot score with numSamples=$numSamples; expected numSamples >= 2.",
    )
    require(
      isolationTrees.nonEmpty,
      "Cannot score with an empty IsolationForestModel.",
    )
    transformSchema(data.schema, logging = true)

    val avgPath = avgPathLength(numSamples)
    val broadcastIsolationTrees = data.sparkSession.sparkContext.broadcast(isolationTrees)

    val calculatePathLength = (features: Vector) => {
      if (hasKnownTotalNumFeatures) {
        validateFeatureVectorSize(features, totalNumFeatures)
      }
      val pathLength = broadcastIsolationTrees.value
        .map(y => y.calculatePathLength(DataPoint(features.toArray.map(x => x.toFloat))))
        .sum / broadcastIsolationTrees.value.length
      Math.pow(2, -pathLength / avgPath)
    }
    val transformUDF = udf(calculatePathLength)

    val dataWithScores = data.withColumn($(scoreCol), transformUDF(col($(featuresCol))))
    val dataWithScoresAndPrediction = if (outlierScoreThreshold > 0) {
      dataWithScores
        .withColumn($(predictionCol), (col($(scoreCol)) >= outlierScoreThreshold).cast("double"))
    } else {
      dataWithScores.withColumn($(predictionCol), lit(0.0))
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
  override def write: MLWriter = new IsolationForestModelReadWrite.IsolationForestModelWriter(this)
}

/**
 * Companion object to the IsolationForestModel class.
 */
case object IsolationForestModel extends MLReadable[IsolationForestModel] {

  private[isolationforest] val UnknownTotalNumFeatures: Int = -1

  /**
   * Returns an IsolationForestModelReader instance that can be used to read a saved isolation
   * forest from disk.
   *
   * @return
   *   An IsolationForestModelReader instance.
   */
  override def read: MLReader[IsolationForestModel] =
    new IsolationForestModelReadWrite.IsolationForestModelReader

  /**
   * Loads a saved isolation forest model from disk. A shortcut of `read.load(path)`.
   *
   * @param path
   *   The path to the saved isolation forest model.
   * @return
   *   The loaded IsolationForestModel instance.
   */
  override def load(path: String): IsolationForestModel = super.load(path)
}
