package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.SharedTrainLogic.{
  computeAndSetModelThreshold,
  createSampledPartitionedDataset,
  trainIsolationTrees,
}
import com.linkedin.relevance.isolationforest.core.Utils.{DataPoint, ResolvedParams}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
 * Used to train an extended isolation forest model. It extends the spark.ml Estimator class.
 *
 * This uses random hyperplanes (rather than single-feature splits) to isolate outliers.
 */
class ExtendedIsolationForest(override val uid: String)
    extends Estimator[ExtendedIsolationForestModel]
    with ExtendedIsolationForestParams
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("extended-isolation-forest"))

  override def copy(extra: ParamMap): ExtendedIsolationForest =
    copyValues(new ExtendedIsolationForest(uid), extra)

  /**
   * Fits an extended isolation forest model given an input dataset of features.
   *
   * @param data
   *   The input dataset, which must contain a column $(featuresCol) of Vectors.
   */
  override def fit(data: Dataset[_]): ExtendedIsolationForestModel = {

    import data.sparkSession.implicits._

    // Validate schema, extract features column, and convert to Dataset
    transformSchema(data.schema, logging = true)

    val df = data.toDF()
    val dataset =
      df.map(row => DataPoint(row.getAs[Vector]($(featuresCol)).toArray.map(x => x.toFloat)))

    // Validate $(maxFeatures) and $(maxSamples) against input dataset and determine the values
    // actually used to train the model: numFeatures and numSamples
    val resolvedParams = validateAndResolveParams(dataset)

    // Bag and flatten the data, then repartition it so that each partition corresponds to one
    // isolation tree.
    val repartitionedFlattenedSampledDataset = createSampledPartitionedDataset(
      dataset,
      resolvedParams.numSamples,
      resolvedParams.totalNumSamples,
      $(numEstimators),
      $(bootstrap),
      $(randomSeed),
    )

    // Train an isolation tree on each subset of data.
    val extendedIsolationTrees = trainIsolationTrees[ExtendedIsolationTree](
      repartitionedFlattenedSampledDataset,
      resolvedParams.numSamples,
      resolvedParams.numFeatures,
      $(randomSeed) + 2 * (dataset.rdd.getNumPartitions + 1),
      (dataArray: Array[DataPoint], seed: Long, featureIndices: Array[Int]) =>
        // We'll define an inline function that calls ExtendedIsolationTree.fit with extensionLevel
        ExtendedIsolationTree.fit(dataArray, seed, featureIndices, $(extensionLevel)),
    )

    // Create the ExtendedIsolationForestModel instance and set the parent.
    val extendedIsolationForestModel = copyValues(
      new ExtendedIsolationForestModel(
        uid,
        extendedIsolationTrees,
        resolvedParams.numSamples,
        resolvedParams.numFeatures,
      ).setParent(this),
    )

    // Determine and set the model threshold based upon the specified contamination and
    // contaminationError parameters.
    computeAndSetModelThreshold(
      extendedIsolationForestModel,
      df,
      $(scoreCol),
      $(contamination),
      $(contaminationError),
    )

    extendedIsolationForestModel
  }

  /**
   * Validates the input schema and transforms it into the output schema. It validates that the
   * input DataFrame has a $(featuresCol) of the correct type and appends the output columns to the
   * input schema. It also ensures that the input DataFrame does not already have $(predictionCol)
   * or $(scoreCol) columns, as they will be created during the fitting process.
   *
   * @param schema
   *   The schema of the DataFrame containing the data to be fit.
   * @return
   *   The schema of the DataFrame containing the data to be fit, with the additional
   *   $(predictionCol) and $(scoreCol) columns added.
   */
  override def transformSchema(schema: StructType): StructType = {

    require(
      schema.fieldNames.contains($(featuresCol)),
      s"Input column ${$(featuresCol)} does not exist.",
    )
    require(
      schema($(featuresCol)).dataType == VectorType,
      s"Input column ${$(featuresCol)} is not of required type ${VectorType}",
    )

    require(
      !schema.fieldNames.contains($(predictionCol)),
      s"Output column ${$(predictionCol)} already exists.",
    )
    require(
      !schema.fieldNames.contains($(scoreCol)),
      s"Output column ${$(scoreCol)} already exists.",
    )

    val outputFields = schema.fields :+
      StructField($(predictionCol), DoubleType, nullable = false) :+
      StructField($(scoreCol), DoubleType, nullable = false)

    StructType(outputFields)
  }

  /**
   * Private helper to validate parameters and figure out how many features and samples we'll use.
   *
   * @param dataset
   *   The input dataset.
   * @return
   *   A ResolvedParams instance containing the resolved values.
   */
  private def validateAndResolveParams(dataset: Dataset[DataPoint]): ResolvedParams = {

    // Validate $(maxFeatures) and $(maxSamples) against input dataset and determine the values
    // actually used to train the model: numFeatures and numSamples.
    val totalNumFeatures = dataset.head().features.length
    val numFeatures = if ($(maxFeatures) > 1.0) {
      math.floor($(maxFeatures)).toInt
    } else {
      math.floor($(maxFeatures) * totalNumFeatures).toInt
    }
    logInfo(
      s"User specified number of features used to train each tree over total number of" +
        s" features: ${numFeatures} / ${totalNumFeatures}",
    )
    require(
      numFeatures > 0,
      s"parameter maxFeatures given invalid value ${$(maxFeatures)}" +
        s" specifying the use of ${numFeatures} features, but >0 features are required.",
    )
    require(
      numFeatures <= totalNumFeatures,
      s"parameter maxFeatures given invalid value" +
        s" ${$(maxFeatures)} specifying the use of ${numFeatures} features, but only" +
        s" ${totalNumFeatures} features are available.",
    )

    val totalNumSamples = dataset.count()
    val numSamples = if ($(maxSamples) > 1.0) {
      math.floor($(maxSamples)).toInt
    } else {
      math.floor($(maxSamples) * totalNumSamples).toInt
    }
    logInfo(
      s"User specified number of samples used to train each tree over total number of" +
        s" samples: ${numSamples} / ${totalNumSamples}",
    )
    require(
      numSamples > 0,
      s"parameter maxSamples given invalid value ${$(maxSamples)}" +
        s" specifying the use of ${numSamples} samples, but >0 samples are required.",
    )
    require(
      numSamples <= totalNumSamples,
      s"parameter maxSamples given invalid value" +
        s" ${$(maxSamples)} specifying the use of ${numSamples} samples, but only" +
        s" ${totalNumSamples} samples are in the input dataset.",
    )

    ResolvedParams(numFeatures, totalNumFeatures, numSamples, totalNumSamples)
  }
}

/**
 * Companion object to the ExtendedIsolationForest class.
 */
object ExtendedIsolationForest extends DefaultParamsReadable[ExtendedIsolationForest] {

  /**
   * Loads a saved ExtendedIsolationForest Estimator ML instance.
   *
   * @param path
   *   Path to the saved ExtendedIsolationForest Estimator ML instance directory.
   * @return
   *   The saved ExtendedIsolationForest Estimator ML instance.
   */
  override def load(path: String): ExtendedIsolationForest = super.load(path)
}
