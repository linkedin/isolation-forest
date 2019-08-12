package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Utils.{avgPathLength, DataPoint}
import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}


/**
  * A trained isolation tree model. It extends the spark.ml Model class.
  *
  * @param uid The immutable unique ID for the model.
  * @param isolationTrees The array of isolation tree models that compose the isolation forest.
  */
class IsolationForestModel(
  override val uid: String,
  val isolationTrees: Array[IsolationTree],
  private val numSamples: Int)
  extends Model[IsolationForestModel] with IsolationForestParams with MLWritable {

  require(numSamples > 0, s"parameter numSamples must be >0, but given invalid value ${numSamples}")
  final def getNumSamples: Int = numSamples

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

  override def copy(extra: ParamMap): IsolationForestModel = {

    val isolationForestCopy = new IsolationForestModel(uid, isolationTrees, numSamples)
      .setParent(this.parent)
    isolationForestCopy.setOutlierScoreThreshold(outlierScoreThreshold)
    copyValues(isolationForestCopy, extra)
  }

  /**
    * Scores new data instances using the trained isolation forest model.
    *
    * @param data The input DataFrame of data to be scored. It must have a column $(featuresCol)
    *             that contains a the feature vector for each data instance.
    * @return The same DataFrame with $(predictionCol) and $(scoreCol) appended.
    */
  override def transform(data: Dataset[_]): DataFrame = {

    transformSchema(data.schema, logging = true)

    val avgPath = avgPathLength(numSamples)
    val broadcastIsolationTrees = data.sparkSession.sparkContext.broadcast(isolationTrees)

    val calculatePathLength = (features: Vector) => {
      val pathLength = broadcastIsolationTrees.value
        .map(y => y.calculatePathLength(DataPoint(features.toArray.map(x => x.toFloat))))
        .sum / $(numEstimators)
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

  /**
    * Validates the input schema and transforms it into the output schema. It validates that the
    * input DataFrame has a $(featuresCol) of the correct type.  It also ensures that the input
    * DataFrame does not already have $(predictionCol) or $(scoreCol) columns, as they will be
    * created during the fitting process.
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
  override def write: MLWriter = new IsolationForestModelReadWrite.IsolationForestModelWriter(this)
}

/**
  * Companion object to the IsolationForestModel class.
  */
case object IsolationForestModel extends MLReadable[IsolationForestModel] {

  /**
    * Returns an IsolationForestModelReader instance that can be used to read a saved isolation
    * forest from disk.
    *
    * @return An IsolationForestModelReader instance.
    */
  override def read: MLReader[IsolationForestModel] =
    new IsolationForestModelReadWrite.IsolationForestModelReader

  /**
    * Loads a saved isolation forest model from disk. A shortcut of `read.load(path)`.
    *
    * @param path The path to the saved isolation forest model.
    * @return The loaded IsolationForestModel instance.
    */
  override def load(path: String): IsolationForestModel = super.load(path)
}
