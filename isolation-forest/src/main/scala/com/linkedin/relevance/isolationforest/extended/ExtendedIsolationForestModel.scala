package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.Utils.{DataPoint, avgPathLength}
import org.apache.spark.ml.{Model}
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A trained extended isolation forest model, composed of multiple ExtendedIsolationTree
 * instances that each use random hyperplane splits.
 *
 * @param uid            Unique ID for the model instance.
 * @param isolationTrees The array of extended isolation trees in this ensemble.
 * @param numSamples     The number of samples used to train each tree.
 * @param numFeatures    The number of features used in each tree's hyperplane subspace.
 */
class ExtendedIsolationForestModel(
  override val uid: String,
  val isolationTrees: Array[ExtendedIsolationTree],
  private val numSamples: Int,
  private val numFeatures: Int) extends Model[ExtendedIsolationForestModel] with ExtendedIsolationForestParams with MLWritable {

  require(numSamples > 0, s"param numSamples must be > 0 but found $numSamples")
  require(numFeatures > 0, s"param numFeatures must be > 0 but found $numFeatures")

  // This threshold is set later by computeAndSetModelThreshold if contamination > 0
  private var outlierScoreThreshold: Double = -1.0

  private[isolationforest] def setOutlierScoreThreshold(value: Double): Unit = {
    require(value == -1.0 || (value >= 0.0 && value <= 1.0),
      s"outlierScoreThreshold must be in [0,1] or -1, but got $value")
    outlierScoreThreshold = value
  }

  def getOutlierScoreThreshold: Double = outlierScoreThreshold
  def getNumSamples: Int = numSamples
  def getNumFeatures: Int = numFeatures

  override def copy(extra: ParamMap): ExtendedIsolationForestModel = {
    val copied = new ExtendedIsolationForestModel(uid, isolationTrees, numSamples, numFeatures)
      .setParent(parent)
    copyValues(copied, extra)
    copied.setOutlierScoreThreshold(outlierScoreThreshold)
    copied
  }

  /**
   * Transform: compute outlier scores for each row, then optionally produce a predicted label
   * if outlierScoreThreshold >= 0.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val avgPath = avgPathLength(numSamples)
    val broadcastTrees = dataset.sparkSession.sparkContext.broadcast(isolationTrees)
    val calcScore = udf { features: Vector =>
      val pathLengthSum = broadcastTrees.value.map { tree =>
        tree.calculatePathLength(DataPoint(features.toArray.map(_.toFloat)))
      }.sum
      val meanPathLength = pathLengthSum / $(numEstimators)
      // final anomaly score = 2^(-meanPathLength / avgPath)
      math.pow(2.0, -meanPathLength / avgPath)
    }

    val dfWithScore = dataset.withColumn($(scoreCol), calcScore(col($(featuresCol))))
    if (outlierScoreThreshold > 0.0) {
      dfWithScore.withColumn(
        $(predictionCol),
        (col($(scoreCol)) >= outlierScoreThreshold).cast("double")
      )
    } else {
      // If threshold is not set, default predictedLabel to 0.0
      dfWithScore.withColumn($(predictionCol), lit(0.0))
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(featuresCol)),
      s"Input column ${$(featuresCol)} does not exist.")
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType,
      s"Input column ${$(featuresCol)} must be VectorType.")

    require(!schema.fieldNames.contains($(predictionCol)),
      s"Output column ${$(predictionCol)} already exists.")
    require(!schema.fieldNames.contains($(scoreCol)),
      s"Output column ${$(scoreCol)} already exists.")

    val fieldsOut = schema.fields :+
      StructField($(predictionCol), DoubleType, nullable = false) :+
      StructField($(scoreCol), DoubleType, nullable = false)

    StructType(fieldsOut)
  }

  /**
   * (Optional) Implement a custom writer if you need saving/loading support
   * for ExtendedIsolationForestModel, mirroring `IsolationForestModelReadWrite`.
   */
  override def write: MLWriter = {
    throw new UnsupportedOperationException(
      "ExtendedIsolationForestModel.write is not implemented. " +
        "Implement a custom read/write class if serialization is required."
    )
  }
}

/**
 * Companion object for ExtendedIsolationForestModel, if you wish to implement
 * read/load. Mirroring how standard IsolationForestModel is done.
 */
object ExtendedIsolationForestModel {
  // If you want to implement read/save:
  // def read: MLReader[ExtendedIsolationForestModel] = ...
  // def load(path: String): ExtendedIsolationForestModel = ...
}
