package com.linkedin.relevance.isolationforest.extended

import com.linkedin.relevance.isolationforest.core.SharedTrainLogic.{
  computeAndSetModelThreshold,
  createSampledPartitionedDataset,
  trainIsolationTrees,
  validateAndResolveParams,
}
import com.linkedin.relevance.isolationforest.core.Utils.{DataPoint, validateAndTransformSchema}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

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
    val resolvedParams = validateAndResolveParams(dataset, $(maxFeatures), $(maxSamples))

    // Resolve the effective extension level: default to fully extended if not explicitly set,
    // otherwise validate the user-specified value against the resolved feature subspace dimension.
    val maxExtensionLevel = resolvedParams.numFeatures - 1
    val resolvedExtensionLevel = if (isSet(extensionLevel)) {
      require(
        $(extensionLevel) <= maxExtensionLevel,
        s"parameter extensionLevel given invalid value ${$(extensionLevel)}," +
          s" but must be in [0, $maxExtensionLevel] for a subspace of" +
          s" ${resolvedParams.numFeatures} features.",
      )
      $(extensionLevel)
    } else {
      maxExtensionLevel
    }
    logInfo(s"Using extensionLevel=$resolvedExtensionLevel (max=$maxExtensionLevel)")

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
        ExtendedIsolationTree.fit(dataArray, seed, featureIndices, resolvedExtensionLevel),
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
    extendedIsolationForestModel.set(extensionLevel, resolvedExtensionLevel)

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

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, $(featuresCol), $(predictionCol), $(scoreCol))

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
