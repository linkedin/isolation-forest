package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.SharedTrainLogic.{
  computeAndSetModelThreshold,
  createSampledPartitionedDataset,
  trainIsolationTrees
}
import com.linkedin.relevance.isolationforest.core.Utils
import com.linkedin.relevance.isolationforest.core.Utils.DataPoint
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
 * Estimator for an Extended Isolation Forest.
 *
 * This uses random hyperplanes (rather than single-feature splits) to isolate outliers.
 */
class ExtendedIsolationForest(override val uid: String)
  extends Estimator[ExtendedIsolationForestModel]
    with ExtendedIsolationForestParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("extended-isolation-forest"))

  override def copy(extra: ParamMap): ExtendedIsolationForest = {
    copyValues(new ExtendedIsolationForest(uid), extra)
  }

  /**
   * Fits an extended isolation forest model given an input dataset of features.
   *
   * @param dataset The input dataset, which must contain a column $(featuresCol) of ML vectors.
   */
  override def fit(dataset: Dataset[_]): ExtendedIsolationForestModel = {

    transformSchema(dataset.schema, logging = true)

    import dataset.sparkSession.implicits._

    val df = dataset.toDF()
    val points = df.map(row => DataPoint(row.getAs[Vector]($(featuresCol)).toArray.map(_.toFloat)))

    // Validate user params and figure out how many samples & features to actually use
    val resolvedParams = validateAndResolveParams(points)

    // Bagging: sample data for each tree
    val partitionedDataset = createSampledPartitionedDataset(
      points,
      resolvedParams.numSamples,
      resolvedParams.totalNumSamples,
      $(numEstimators),
      $(bootstrap),
      $(randomSeed)
    )

    // Train extended isolation trees
    val trees = trainIsolationTrees[ExtendedIsolationTree](
      partitionedDataset,
      resolvedParams.numSamples,
      resolvedParams.numFeatures,
      $(randomSeed) + 10 * $(numEstimators),
      (dataArray: Array[DataPoint], seed: Long, featureIndices: Array[Int]) => {
        // We'll define an inline function that calls ExtendedIsolationTree.fit with extensionLevel
        ExtendedIsolationTree.fit(dataArray, seed, featureIndices, $(extensionLevel))
      }
    )

    val model = copyValues(
      new ExtendedIsolationForestModel(
        uid,
        trees,
        resolvedParams.numSamples,
        resolvedParams.numFeatures
      ).setParent(this)
    )

    // If contamination > 0, approximate a threshold
    computeAndSetModelThreshold(
      model,
      df,
      $(scoreCol),
      $(contamination),
      $(contaminationError)
    )

    model
  }

  /**
   * Validate the input schema, ensuring $(featuresCol) is present and is a Vector.
   * Ensure that $(predictionCol) and $(scoreCol) do not already exist.
   */
  override def transformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(featuresCol)),
      s"Input column ${$(featuresCol)} does not exist.")
    require(schema($(featuresCol)).dataType == SQLDataTypes.VectorType,
      s"Input column ${$(featuresCol)} must be VectorType.")

    require(!schema.fieldNames.contains($(predictionCol)),
      s"Output column ${$(predictionCol)} already exists.")
    require(!schema.fieldNames.contains($(scoreCol)),
      s"Output column ${$(scoreCol)} already exists.")

    val fieldsPlus = schema.fields :+
      StructField($(predictionCol), DoubleType, nullable = false) :+
      StructField($(scoreCol), DoubleType, nullable = false)

    StructType(fieldsPlus)
  }

  /**
   * Helper to figure out how many samples/features to use from the dataset
   * given the user-specified maxSamples/maxFeatures.
   */
  private def validateAndResolveParams(points: Dataset[DataPoint]): Utils.ResolvedParams = {
    val totalNumFeatures = points.head().features.length
    val totalNumSamples = points.count()

    val actualNumFeatures =
      if (getMaxFeatures > 1.0) math.floor(getMaxFeatures).toInt
      else math.floor(getMaxFeatures * totalNumFeatures).toInt

    require(actualNumFeatures > 0 && actualNumFeatures <= totalNumFeatures,
      s"Invalid maxFeatures=${getMaxFeatures}, specifying $actualNumFeatures features, " +
        s"but dataset only has $totalNumFeatures features.")

    val actualNumSamples =
      if (getMaxSamples > 1.0) math.floor(getMaxSamples).toInt
      else math.floor(getMaxSamples * totalNumSamples).toInt

    require(actualNumSamples > 0 && actualNumSamples <= totalNumSamples,
      s"Invalid maxSamples=${getMaxSamples}, specifying $actualNumSamples samples, " +
        s"but dataset has $totalNumSamples total rows.")

    Utils.ResolvedParams(
      numFeatures = actualNumFeatures,
      totalNumFeatures = totalNumFeatures,
      numSamples = actualNumSamples,
      totalNumSamples = totalNumSamples
    )
  }
}

object ExtendedIsolationForest extends DefaultParamsReadable[ExtendedIsolationForest] {
  override def load(path: String): ExtendedIsolationForest = super.load(path)
}
