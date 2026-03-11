package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.core.IsolationForestParamsBase
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
    val resolvedParams = validateAndResolveParams(dataset, $(maxFeatures), $(maxSamples))

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

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, $(featuresCol), $(predictionCol), $(scoreCol))
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
