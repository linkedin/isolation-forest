package com.linkedin.relevance.isolationforest.core

import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
 * Useful utilities.
 */
private[isolationforest] object Utils extends Serializable {

  case class DataPoint(features: Array[Float])
  case class ResolvedParams(
    numFeatures: Int,
    totalNumFeatures: Int,
    numSamples: Int,
    totalNumSamples: Long,
  )
  case class OutlierScore(score: Double)

  /**
   * Validates the input schema and appends the output columns. Checks that the input DataFrame has
   * a featuresCol of the correct type and does not already have predictionCol or scoreCol columns.
   *
   * @param schema
   *   The schema of the input DataFrame.
   * @param featuresCol
   *   The name of the features column.
   * @param predictionCol
   *   The name of the prediction column to be added.
   * @param scoreCol
   *   The name of the score column to be added.
   * @return
   *   The input schema with the additional predictionCol and scoreCol columns appended.
   */
  def validateAndTransformSchema(
    schema: StructType,
    featuresCol: String,
    predictionCol: String,
    scoreCol: String,
  ): StructType = {

    require(
      schema.fieldNames.contains(featuresCol),
      s"Input column ${featuresCol} does not exist.",
    )
    require(
      schema(featuresCol).dataType == SQLDataTypes.VectorType,
      s"Input column ${featuresCol} is not of required type ${SQLDataTypes.VectorType}",
    )

    require(
      !schema.fieldNames.contains(predictionCol),
      s"Output column ${predictionCol} already exists.",
    )
    require(
      !schema.fieldNames.contains(scoreCol),
      s"Output column ${scoreCol} already exists.",
    )

    val outputFields = schema.fields :+
      StructField(predictionCol, DoubleType, nullable = false) :+
      StructField(scoreCol, DoubleType, nullable = false)

    StructType(outputFields)
  }

  def validateFeatureVectorSize(features: Vector, expectedNumFeatures: Int): Unit =
    require(
      features.size == expectedNumFeatures,
      s"Input feature vector size ${features.size} did not match the model's training dimension" +
        s" ${expectedNumFeatures}.",
    )

  val EulerConstant = 0.5772156649f

  /**
   * Returns the average path length for an unsuccessful BST search. It is Equation 1 in the 2008
   * "Isolation Forest" paper by F. T. Liu, et al.
   *
   * @param numInstances
   *   The number of data points in the root node of the BST.
   * @return
   *   The average path length of an unsuccessful BST search.
   */
  def avgPathLength(numInstances: Long): Float =

    if (numInstances <= 1) {
      0.0f
    } else {
      2.0f * (math.log(numInstances.toFloat - 1.0f).toFloat + EulerConstant) -
        (2.0f * (numInstances.toFloat - 1.0f) / numInstances.toFloat)
    }
}
