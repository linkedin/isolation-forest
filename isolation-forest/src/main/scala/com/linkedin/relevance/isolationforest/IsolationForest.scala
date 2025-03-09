package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.IsolationForestParamsBase
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
 * Used to train an isolation forest model. It extends the spark.ml Estimator class.
 *
 * @param uid
 *   The immutable unique ID for the model.
 */
class IsolationForest(override val uid: String)
    extends Estimator[IsolationForestModel]
    with IsolationForestParamsBase
    with DefaultParamsWritable
    with Logging {

  def this() = this(Identifiable.randomUID("isolation-forest"))

  override def copy(extra: ParamMap): IsolationForest =

    copyValues(new IsolationForest(uid), extra)

  /**
   * Fits an isolation forest given an input DataFrame.
   *
   * @param data
   *   A DataFrame with a column $(featuresCol) that contains a the feature vector for each data
   *   instance.
   * @return
   *   The trained isolation forest model.
   */
  override def fit(data: Dataset[_]): IsolationForestModel = {

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
    val isolationTrees = trainIsolationTrees[IsolationTree](
      repartitionedFlattenedSampledDataset,
      resolvedParams.numSamples,
      resolvedParams.numFeatures,
      $(
        randomSeed,
      ) + 2 * (dataset.rdd.getNumPartitions + 1), // Offset the random seed to avoid collisions.
      treeBuilder = IsolationTree.fit,
    )

    // Create the IsolationForestModel instance and set the parent.
    val isolationForestModel = copyValues(
      new IsolationForestModel(
        uid,
        isolationTrees,
        resolvedParams.numSamples,
        resolvedParams.numFeatures,
      )
        .setParent(this),
    )

    // Determine and set the model threshold based upon the specified contamination and
    // contaminationError parameters.
    computeAndSetModelThreshold(
      isolationForestModel,
      df,
      $(scoreCol),
      $(contamination),
      $(contaminationError),
    )

    isolationForestModel
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
}

/**
 * Companion object to the IsolationForest class.
 */
case object IsolationForest extends DefaultParamsReadable[IsolationForest] {

  /**
   * Loads a saved IsolationForest Estimator ML instance.
   *
   * @param path
   *   Path to the saved IsolationForest Estimator ML instance directory.
   * @return
   *   The saved IsolationForest Estimator ML instance.
   */
  override def load(path: String): IsolationForest = super.load(path)
}
