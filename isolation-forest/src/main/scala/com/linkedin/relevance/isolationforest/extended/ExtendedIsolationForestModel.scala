package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.{IsolationForestModel, IsolationForestModelReadWrite}
import com.linkedin.relevance.isolationforest.core.Utils.{DataPoint, avgPathLength}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A trained extended isolation forest model, composed of multiple ExtendedIsolationTree
 * instances that each use random hyperplane splits.
 *
 * @param uid                    The immutable unique ID for the model.
 * @param extendedIsolationTrees The array of extended isolation tree models that compose
 *                               the extended isolation forest.
 * @param numSamples             The number of samples used to train each tree.
 * @param numFeatures            The number of features used in each tree's hyperplane subspace.
 */
class ExtendedIsolationForestModel(
  override val uid: String,
  val extendedIsolationTrees: Array[ExtendedIsolationTree],
  private val numSamples: Int,
  private val numFeatures: Int) extends Model[ExtendedIsolationForestModel] with ExtendedIsolationForestParams with MLWritable {

  require(numSamples > 0, s"parameter numSamples must be >0, but given invalid value ${numSamples}")
  final def getNumSamples: Int = numSamples

  require(numFeatures > 0, s"parameter numFeatures must be >0, but given invalid value ${numFeatures}")
  final def getNumFeatures: Int = numFeatures

  // The outlierScoreThreshold needs to be a mutable variable because it is not known when an
  // IsolationForestModel instance is created.
  private var outlierScoreThreshold: Double = -1
  private[isolationforest] def setOutlierScoreThreshold(value: Double): Unit = {

    require(value == -1 || (value >= 0 && value <= 1), "parameter outlierScoreThreshold must be" +
      " equal to -1 (no threshold) or be in the range [0, 1]," +
      s" but given invalid value ${value}")
    outlierScoreThreshold = value
  }
  final def getOutlierScoreThreshold: Double = outlierScoreThreshold

  override def copy(extra: ParamMap): ExtendedIsolationForestModel = {
    val extendedIsolationForestCopy = new ExtendedIsolationForestModel(
      uid,
      extendedIsolationTrees,
      numSamples,
      numFeatures
    ).setParent(this.parent)
    extendedIsolationForestCopy.setOutlierScoreThreshold(outlierScoreThreshold)

    copyValues(extendedIsolationForestCopy, extra)
  }

  /**
   * Scores new data instances using the trained extended isolation forest model.
   *
   * @param data The input DataFrame of data to be scored. It must have a column $(featuresCol)
   *             that contains a feature vector for each data instance.
   * @return The same DataFrame with $(predictionCol) and $(scoreCol) appended.
   */
  override def transform(data: Dataset[_]): DataFrame = {

    transformSchema(data.schema, logging = true)

    val avgPath = avgPathLength(numSamples)
    val broadcastExtendedIsolationTrees = data.sparkSession.sparkContext
      .broadcast(extendedIsolationTrees)

    val calculatePathLength = (features: Vector) => {
      val pathLength = broadcastExtendedIsolationTrees.value
        .map(y => y.calculatePathLength(DataPoint(features.toArray.map(x => x.toFloat))))
        .sum / $(numEstimators)
      Math.pow(2, -pathLength / avgPath)
    }
    val transformUDF = udf(calculatePathLength)

    val dataWithScore = data.withColumn($(scoreCol), transformUDF(col($(featuresCol))))
    val dataWithScoresAndPrediction = if (outlierScoreThreshold > 0.0) {
      dataWithScore.withColumn(
        $(predictionCol),
        (col($(scoreCol)) >= outlierScoreThreshold).cast("double")
      )
    } else {
      // If threshold is not set, default predictedLabel to 0.0
      dataWithScore.withColumn($(predictionCol), lit(0.0))
    }

    dataWithScoresAndPrediction
  }

  /**
   * Validates the input schema and transforms it into the output schema. It validates that the
   * input DataFrame has a $(featuresCol) of the correct type and appends the output columns to
   * the input schema. It also ensures that the input DataFrame does not already have
   * $(predictionCol) or $(scoreCol) columns, as they will be created during the fitting process.
   *
   * @param schema The schema of the DataFrame containing the data to be fit.
   * @return The schema of the DataFrame containing the data to be fit, with the additional
   *         $(predictionCol) and $(scoreCol) columns added.
   */
  override def transformSchema(schema: StructType): StructType = {

    require(schema.fieldNames.contains($(featuresCol)),
      s"Input column ${$(featuresCol)} does not exist.")
    require(schema($(featuresCol)).dataType == VectorType,
      s"Input column ${$(featuresCol)} is not of required type ${VectorType}")

    require(!schema.fieldNames.contains($(predictionCol)),
      s"Output column ${$(predictionCol)} already exists.")
    require(!schema.fieldNames.contains($(scoreCol)),
      s"Output column ${$(scoreCol)} already exists.")

    val outputFields = schema.fields :+
      StructField($(predictionCol), DoubleType, nullable = false) :+
      StructField($(scoreCol), DoubleType, nullable = false)

    StructType(outputFields)
  }

  /**
   * Returns an IsolationForestModelWriter instance that can be used to write the isolation forest
   * to disk.
   *
   * @return An IsolationForestModelWriter instance.
   */
  override def write: MLWriter =
    new ExtendedIsolationForestModelReadWrite.ExtendedIsolationForestModelWriter(this)
}

/**
 * Companion object to the IsolationForestModel class.
 */
case object ExtendedIsolationForestModel extends MLReadable[ExtendedIsolationForestModel] {

  /**
   * Returns an ExtendedIsolationForestModelReader instance that can be used
   * to read a saved extended isolation forest from disk.
   *
   * @return An ExtendedIsolationForestModelReader instance.
   */
  override def read: MLReader[ExtendedIsolationForestModel] =
    new ExtendedIsolationForestModelReadWrite.ExtendedIsolationForestModelReader

  /**
   * Loads a saved extended isolation forest model from disk. A shortcut of `read.load(path)`.
   *
   * @param path The path to the saved extended isolation forest model.
   * @return The loaded ExtendedIsolationForestModel instance.
   */
  override def load(path: String): ExtendedIsolationForestModel = super.load(path)
}
