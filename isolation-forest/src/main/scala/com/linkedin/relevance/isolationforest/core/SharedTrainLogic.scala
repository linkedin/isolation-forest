package com.linkedin.relevance.isolationforest.core

import com.linkedin.relevance.isolationforest.core.Utils.{DataPoint, OutlierScore}
import org.apache.spark.{HashPartitioner, TaskContext}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD

import scala.language.reflectiveCalls
import scala.reflect.ClassTag

private[isolationforest] object SharedTrainLogic extends Logging {

  /**
   * Helper that bags the data (with possible up-sampling), flattens it, repartitions by tree index,
   * and returns the partitioned Dataset.
   *
   * @param dataset
   *   The input data, in DataPoint form
   * @param numSamples
   *   Number of samples per tree
   * @param totalNumSamples
   *   The total number of data points in `dataset`
   * @param numEstimators
   *   Number of trees to train
   * @param bootstrap
   *   Whether to sample with replacement
   * @param randomSeed
   *   Random seed for sampling
   * @return
   *   A Dataset[DataPoint] partitioned so that each partition corresponds to one isolation tree
   */
  def createSampledPartitionedDataset(
    dataset: Dataset[DataPoint],
    numSamples: Long,
    totalNumSamples: Long,
    numEstimators: Int,
    bootstrap: Boolean,
    randomSeed: Long,
  ): Dataset[DataPoint] = {

    import dataset.sparkSession.implicits._

    // Sample and assign data into $(numEstimators) subsets. Repartition RDD to ensure that the data
    // for each tree is on its own partition; later steps preserve partitioning. We sample more than
    // needed to avoid having too few samples due to random chance. We are typically in the large n,
    // small p (binomial) regime, so we can approximate using a Poisson distribution. We set
    // targetNumSamples to be a 7 sigma up-fluctuation on the number of samples we eventually plan
    // to draw.
    val nSigma = 7.0
    val targetNumSamples = numSamples.toDouble + nSigma * math.sqrt(numSamples.toDouble)
    logInfo(
      s"Expectation value for samples drawn for each tree is ${targetNumSamples} samples." +
        s" This subsample is later limited to user specified ${numSamples} samples before training.",
    )
    val sampleFraction = Math.min(targetNumSamples / totalNumSamples.toDouble, 1.0)
    logInfo(
      s"The subsample for each partition is sampled from input data using sampleFraction =" +
        s" ${sampleFraction}.",
    )

    // Bag and flatten
    val sampledRdd = BaggedPoint.convertToBaggedRDD(
      dataset.rdd,
      sampleFraction,
      numEstimators,
      bootstrap,
      randomSeed,
    )
    val flattenedSampledRdd =
      BaggedPoint.flattenBaggedRDD(sampledRdd, randomSeed + dataset.rdd.getNumPartitions + 1)

    // Partition by tree index
    val repartitionedFlattenedSampledRdd =
      flattenedSampledRdd.partitionBy(new HashPartitioner(numEstimators))

    val repartitionedFlattenedSampledDataset = repartitionedFlattenedSampledRdd
      .mapPartitions(x => x.map(y => y._2), preservesPartitioning = true)
      .toDS()

    logInfo(
      s"Training ${numEstimators} isolation trees using" +
        s" ${repartitionedFlattenedSampledDataset.rdd.getNumPartitions} partitions.",
    )

    repartitionedFlattenedSampledDataset
  }

  /**
   * Helper that computes and sets the model threshold based on the current contamination and
   * contaminationError parameters. It's generic for any model that: 1) Has transform() returning a
   * DF with scoreCol 2) Exposes a method to set threshold
   *
   * If contamination == 0.0, the threshold is not set (no outliers). Otherwise, it uses
   * approxQuantile on the training data to match the desired contamination fraction, and logs a
   * warning if the observed contamination differs more than expected.
   *
   * @param model
   *   The trained isolation forest model instance.
   * @param data
   *   The training data used to train the model.
   * @param scoreCol
   *   The name of the column in the DataFrame that contains the outlier score.
   * @param contamination
   *   The user-specified contamination fraction.
   * @param contaminationError
   *   The user-specified error tolerance for the contamination fraction.
   */
  def computeAndSetModelThreshold(
    model: {
      def transform(data: Dataset[_]): DataFrame; def setOutlierScoreThreshold(v: Double): Unit
    },
    data: Dataset[_],
    scoreCol: String,
    contamination: Double,
    contaminationError: Double,
  ): Unit = {

    import data.sparkSession.implicits._

    if (contamination > 0.0) {
      // Score all training instances to determine the threshold required to achieve the desired
      // level of contamination. The approxQuantile method uses the algorithm in this paper:
      // https://dl.acm.org/citation.cfm?id=375670
      val scores = model
        .transform(data)
        .map(row => OutlierScore(row.getAs[Double](scoreCol)))
        .cache()
      val outlierScoreThreshold = scores.stat
        .approxQuantile("score", Array(1 - contamination), contaminationError)
        .head
      model.setOutlierScoreThreshold(outlierScoreThreshold)

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
        .map(outlierScore => if (outlierScore.score >= outlierScoreThreshold) 1.0 else 0.0)
        .reduce(_ + _) / scores.count()

      // Determine allowable verification range
      val verificationError = if (contaminationError == 0.0) {
        // If the threshold is calculated exactly, then assume a relative 1% error on the specified
        // contamination for the verification.
        contamination * 0.01
      } else {
        contaminationError
      }

      if (math.abs(observedContamination - contamination) > verificationError) {
        logWarning(
          s"Observed contamination is ${observedContamination}, which is outside" +
            s" the expected range of ${contamination} +/- ${verificationError}. If this is" +
            s" acceptable to you, then it is OK to proceed. If there is a very large discrepancy" +
            s" between observed and expected values, then please try retraining the model with an" +
            s" exact threshold calculation (set the contaminationError parameter value to 0.0).",
        )
      }
    } else {
      // Do not set the outlier score threshold, which ensures no outliers are found. This speeds up
      // the algorithm runtime by avoiding the approxQuantile calculation.
      logInfo(
        s"Contamination parameter was set to ${contamination}, so all predicted" +
          " labels will be false. The model and outlier scores are otherwise not affected by this" +
          " parameter choice.",
      )
    }
  }

  /**
   * Helper that trains an isolation tree on each partition of the given Dataset, where each
   * partition corresponds to a single tree.
   *
   * Data for each tree is randomly shuffled before slice to ensure no bias when we limit to
   * numSamples. A unique random seed is used to build each tree.
   *
   * @param dataset
   *   A partitioned Dataset[DataPoint], where each partition is dedicated to one tree.
   * @param numSamples
   *   Number of samples to retain per partition/tree.
   * @param numFeatures
   *   Number of features to use per tree.
   * @param randomSeed
   *   Random seed for training the trees.
   * @param treeBuilder
   *   A function that builds an isolation tree.
   * @tparam T
   *   A subtype of IsolationTreeBase.
   * @return
   *   An array of T, one per partition.
   */
  def trainIsolationTrees[T <: IsolationTreeBase: ClassTag](
    dataset: Dataset[Utils.DataPoint],
    numSamples: Int,
    numFeatures: Int,
    randomSeed: Long,
    treeBuilder: (Array[Utils.DataPoint], Long, Array[Int]) => T,
  ): Array[T] = {

    val rdd: RDD[Utils.DataPoint] = dataset.rdd // convert Dataset to RDD
    // For each partition, train a single isolation tree
    val isolationTrees = rdd
      .mapPartitions { x =>
        val partitionId = TaskContext.get().partitionId()

        // Use a different seed for each partition to ensure each partition has an independent set of
        // random numbers. This ensures each tree is truly trained independently and doing so has a
        // measurable effect on the results.
        val seed = randomSeed + partitionId
        val rnd = new scala.util.Random(seed)

        // Shuffle, then slice to limit the data
        val dataForTree = rnd.shuffle(x.toSeq).slice(0, numSamples).toArray
        if (dataForTree.length != numSamples)
          logWarning(
            s"Isolation tree with random seed ${seed} is trained using " +
              s"${dataForTree.length} data points instead of user specified ${numSamples}",
          )

        // Randomly choose which features will be used
        val featureIndices = rnd
          .shuffle(0 to dataForTree.head.features.length - 1)
          .toArray
          .take(numFeatures)
          .sorted
        if (featureIndices.length != numFeatures)
          logWarning(
            s"Isolation tree with random seed ${seed} is trained using" +
              s" ${featureIndices.length} features instead of user specified ${numFeatures}",
          )

        // Use a different seed for each partition to ensure each partition has an independent set of
        // random numbers. This ensures each tree is truly trained independently and doing so has a
        // measurable effect on the results.
        val tree: T = treeBuilder(dataForTree, seed, featureIndices)
        Iterator(tree)
      }
      .collect()

    isolationTrees
  }
}
