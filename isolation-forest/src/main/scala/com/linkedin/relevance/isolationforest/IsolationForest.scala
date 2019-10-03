package com.linkedin.relevance.isolationforest

import com.linkedin.relevance.isolationforest.Utils.{DataPoint, OutlierScore}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.Estimator
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Dataset
import org.apache.spark.{HashPartitioner, TaskContext}


/**
  * Used to train an isolation forest model. It extends the spark.ml Estimator class.
  *
  * @param uid The immutable unique ID for the model.
  */
class IsolationForest(override val uid: String) extends Estimator[IsolationForestModel]
  with IsolationForestParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("isolation-forest"))

  override def copy(extra: ParamMap): IsolationForest = {

    copyValues(new IsolationForest(uid), extra)
  }

  /**
    * Fits an isolation forest given an input DataFrame.
    *
    * @param data A DataFrame with a column $(featuresCol) that contains a the feature vector for
    *             each data instance.
    * @return The trained isolation forest model.
    */
  override def fit(data: Dataset[_]): IsolationForestModel = {

    import data.sparkSession.implicits._

    // Validate schema, extract features column, and convert to Dataset
    transformSchema(data.schema, logging = true)
    val df = data.toDF()
    val dataset = df.map(row =>
      DataPoint(row.getAs[Vector]($(featuresCol)).toArray.map(x => x.toFloat)))

    // Validate $(maxFeatures) and $(maxSamples) against input dataset and determine the values
    // actually used to train the model: numFeatures and numSamples
    val totalNumFeatures = dataset.head.features.length
    val numFeatures = if ($(maxFeatures) > 1.0) {
      math.floor($(maxFeatures)).toInt
    } else {
      math.floor($(maxFeatures) * totalNumFeatures).toInt
    }
    logInfo(s"User specified number of features used to train each tree over total number of" +
      s" features: ${numFeatures} / ${totalNumFeatures}")
    require(numFeatures > 0, s"parameter maxFeatures given invalid value ${$(maxFeatures)}" +
      s" specifying the use of ${numFeatures} features, but >0 features are required.")
    require(numFeatures <= totalNumFeatures, s"parameter maxFeatures given invalid value" +
      s" ${$(maxFeatures)} specifying the use of ${numFeatures} features, but only" +
      s" ${totalNumFeatures} features are available.")

    val totalNumSamples = dataset.count()
    val numSamples = if ($(maxSamples) > 1.0) {
      math.floor($(maxSamples)).toInt
    } else {
      math.floor($(maxSamples) * totalNumSamples).toInt
    }
    logInfo(s"User specified number of samples used to train each tree over total number of" +
      s" samples: ${numSamples} / ${totalNumSamples}")
    require(numSamples > 0, s"parameter maxSamples given invalid value ${$(maxSamples)}" +
      s" specifying the use of ${numSamples} samples, but >0 samples are required.")
    require(numSamples <= totalNumSamples, s"parameter maxSamples given invalid value" +
      s" ${$(maxSamples)} specifying the use of ${numSamples} samples, but only" +
      s" ${totalNumSamples} samples are in the input dataset.")

    // Sample and assign data into $(numEstimators) subsets. Repartition RDD to ensure that the data
    // for each tree is on its own partition; later steps preserve partitioning. We sample more than
    // needed to avoid having too few samples due to random chance. We are typically in the large n,
    // small p (binomial) regime, so we can approximate using a Poisson distribution. We set
    // targetNumSamples to be a 7 sigma up-fluctuation on the number of samples we eventually plan
    // to draw.
    val nSigma = 7.0
    val targetNumSamples = numSamples.toDouble + nSigma * math.sqrt(numSamples.toDouble)
    logInfo(s"Expectation value for samples drawn for each tree is ${targetNumSamples} samples." +
      s" This subsample is later limited to user specified ${numSamples} samples before training.")
    val sampleFraction = Math.min(targetNumSamples / totalNumSamples.toDouble, 1.0)
    logInfo(s"The subsample for each partition is sampled from input data using sampleFraction =" +
      s" ${sampleFraction}.")
    val sampledRdd = BaggedPoint
      .convertToBaggedRDD(dataset.rdd, sampleFraction, $(numEstimators), $(bootstrap), $(randomSeed))
    val flattenedSampledRdd = BaggedPoint.flattenBaggedRDD(sampledRdd, $(randomSeed) + 1)
    val repartitionedFlattenedSampledRdd = flattenedSampledRdd
      .partitionBy(new HashPartitioner($(numEstimators)))
    val repartitionedFlattenedSampledDataset = repartitionedFlattenedSampledRdd
      .mapPartitions(x => x.map(y => y._2), preservesPartitioning = true)
      .toDS
    logInfo(s"Training ${$(numEstimators)} isolation trees using" +
      s" ${repartitionedFlattenedSampledDataset.rdd.getNumPartitions} partitions.")

    // Train an isolation tree on each subset of data. Data for each tree is randomly shuffled
    // before slice to ensure no bias when we limit to numSamples. A unique random seed is used to
    // build each tree.
    implicit val isolationTreeEncoder = org.apache.spark.sql.Encoders.kryo[IsolationTree]
    val isolationTrees = repartitionedFlattenedSampledDataset.mapPartitions( x => {
      // Use a different seed for each partition to ensure each partition has an independent set of
      // random numbers. This ensures each tree is truly trained independently and doing so has a
      // measurable effect on the results.
      val seed = $(randomSeed) + TaskContext.get.partitionId() + 2
      val rnd = new scala.util.Random(seed)

      val dataForTree = rnd.shuffle(x.toSeq).slice(0, numSamples).toArray
      if (dataForTree.length != numSamples)
        logWarning(s"Isolation tree with random seed ${seed} is trained using" +
          s" ${dataForTree.length} data points instead of user specified ${numSamples}")

      val featureIndices = rnd.shuffle(0 to dataForTree.head.features.length - 1).toArray
        .take(numFeatures).sorted
      if (featureIndices.length != numFeatures)
        logWarning(s"Isolation tree with random seed ${seed} is trained using" +
          s" ${featureIndices.length} features instead of user specified ${numFeatures}")

      // Use a different seed for each partition to ensure each partition has an independent set of
      // random numbers. This ensures each tree is truly trained independently and doing so has a
      // measurable effect on the results.
      Iterator(IsolationTree
        .fit(dataForTree, $(randomSeed) + $(numEstimators) + TaskContext.get.partitionId() + 2, featureIndices))
    }).collect()

    val isolationForestModel = copyValues(
      new IsolationForestModel(uid, isolationTrees, numSamples).setParent(this))

    // Determine and set the model threshold based upon the specified contamination and
    // contaminationError parameters.
    if ($(contamination) > 0.0) {
      // Score all training instances to determine the threshold required to achieve the desired
      // level of contamination. The approxQuantile method uses the algorithm in this paper:
      // https://dl.acm.org/citation.cfm?id=375670
      val scores = isolationForestModel
        .transform(df)
        .map(row => OutlierScore(row.getAs[Double]($(scoreCol))))
        .cache()
      val outlierScoreThreshold = scores
        .stat.approxQuantile("score", Array(1 - $(contamination)), $(contaminationError))
        .head
      isolationForestModel.setOutlierScoreThreshold(outlierScoreThreshold)

      // Determine labels for each instance using the newly calculated threshold and verify that the
      // fraction of positive labels is in agreement with the user specified contamination. Issue
      // a warning if the observed contamination in the scored training data is outside the expected
      // bounds.
      //
      // If the user specifies a non-zero contaminationError model parameter, then the
      // verificationError used for the verification calculation is equal to the
      // contaminationError parameter value. If the user selects an "exact" calculation of the
      // threshold by setting the parameter contaminationError = 0.0, then assume a
      // verificationError equal to 1% of the contamination parameter value for the validation
      // calculation.
      val observedContamination = scores
        .map(outlierScore => if(outlierScore.score >= outlierScoreThreshold) 1.0 else 0.0)
        .reduce(_ + _) / scores.count()
      val verificationError = if (${contaminationError} == 0.0) {
        // If the threshold is calculated exactly, then assume a relative 1% error on the specified
        // contamination for the verification.
        $(contamination) * 0.01
      } else {
        ${contaminationError}
      }
      if (math.abs(observedContamination - $(contamination)) > verificationError) {

        logWarning(s"Observed contamination is ${observedContamination}, which is outside" +
          s" the expected range of ${${contamination}} +/- ${verificationError}. If this is" +
          s" acceptable to you, then it is OK to proceed. If there is a very large discrepancy" +
          s" between observed and expected values, then please try retraining the model with an" +
          s" exact threshold calculation (set the contaminationError parameter value to 0.0).")
      }
    } else {
      // Do not set the outlier score threshold, which ensures no outliers are found. This speeds up
      // the algorithm runtime by avoiding the approxQuantile calculation.
      logInfo(s"Contamination parameter was set to ${$(contamination)}, so all predicted" +
        " labels will be false. The model and outlier scores are otherwise not affected by this" +
        " parameter choice.")
    }

    isolationForestModel
  }

  /**
    * Validates the input schema and transforms it into the output schema. It validates that the
    * input DataFrame has a $(featuresCol) of the correct type. In this case, the output schema is
    * identical to the input schema.
    *
    * @param schema The schema of the DataFrame containing the data to be fit.
    * @return The schema of the DataFrame containing the data to be fit.
    */
  override def transformSchema(schema: StructType): StructType = {

    require(schema.fieldNames.contains($(featuresCol)),
      s"Input column ${$(featuresCol)} does not exist.")
    require(schema($(featuresCol)).dataType == VectorType,
      s"Input column ${$(featuresCol)} is not of required type ${VectorType}")

    val outputFields = schema.fields

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
    * @param path Path to the saved IsolationForest Estimator ML instance directory.
    * @return The saved IsolationForest Estimator ML instance.
    */
  override def load(path: String): IsolationForest = super.load(path)
}
