/*
 * This file is a heavily modified version of the spark.ml BaggedPoint implementation, which is open
 * sourced under the Apache 2.0 license.
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 */

package com.linkedin.relevance.isolationforest

import org.apache.commons.math3.distribution.{AbstractIntegerDistribution, BinomialDistribution, PoissonDistribution}
import org.apache.spark.rdd.RDD

import scala.util.Random


/**
  * Internal representation of a datapoint which belongs to several subsamples of the same dataset,
  * particularly for bagging (e.g., for random forests).
  *
  * This holds one instance, as well as an array of weights which represent the (weighted)
  * number of times which this instance appears in each subsamplingRate.
  * E.g., (datum, [1, 0, 4]) indicates that there are 3 subsamples of the dataset and that
  * this datum has 1 copy, 0 copies, and 4 copies in the 3 subsamples, respectively.
  *
  * @param datum Data instance.
  * @param subsampleWeights Weight of this instance in each subsampled dataset.
  */
private[isolationforest] case class BaggedPoint[Datum](datum: Datum, subsampleWeights: Array[Double])
  extends Serializable {

  require(subsampleWeights.forall( weight => weight >= 0 ), "All values in subsampleWeights Array" +
    " must be >=0")

  /**
    * This constructor creates a BaggedPoint record, and generates its samples.
    *
    * @param datum Data instance.
    * @param subsamplingRate Fraction of the training data used for learning decision tree.
    * @param numSubsamples Number of subsamples of this RDD to take.
    * @param randomState AbstractIntegerDistribution instance used to draw random samples.
    */
  def this(
    datum: Datum,
    subsamplingRate: Double,
    numSubsamples: Int,
    randomState: AbstractIntegerDistribution) = this(
      datum,
      subsampleWeights = {
        BaggedPoint.enforceRequirements(subsamplingRate, numSubsamples)
        Array.fill(numSubsamples) { randomState.sample() }
      }
  )
}

/**
  * Companion object for BaggedPoint for defining constructors that create the subsampleWeights.
  */
private[isolationforest] case object BaggedPoint {

  /**
    * Enforces the parameter constraints for BaggedPoint constructors.
    *
    * @param subsamplingRate Fraction of the training data used for learning decision tree.
    * @param numSubsamples Number of subsamples of this RDD to take.
    */
  private def enforceRequirements(subsamplingRate: Double, numSubsamples: Int): Unit = {
    require(subsamplingRate > 0 && subsamplingRate <= 1, "parameter subsamplingRate must be in" +
      s" the range 0 < subsamplingRate <= 1, but has value of ${subsamplingRate}.")
    require(numSubsamples > 0, s"parameter numSubsamples must be >0, but has value of" +
      s" ${numSubsamples}.")
  }

  /**
    * Convert an input dataset into its BaggedPoint representation, choosing subsamplingRate counts
    * for each instance.
    *
    * @param input Input dataset.
    * @param subsamplingRate Fraction of the training data used for learning decision tree.
    * @param numSubsamples Number of subsamples of this RDD to take.
    * @param withReplacement Sampling with/without replacement.
    * @param seed Random seed.
    * @return BaggedPoint dataset representation.
    */
  def convertToBaggedRDD[Datum](
    input: RDD[Datum],
    subsamplingRate: Double,
    numSubsamples: Int,
    withReplacement: Boolean,
    seed: Long = Random.nextLong()): RDD[BaggedPoint[Datum]] = {

    /**
      * Convert an input dataset into its BaggedPoint representation, choosing subsamplingRate
      * counts for each instance. This method is a helper for convertToBaggedRDD(). It accepts a
      * randomState parameter of AbstractIntegerDistribution type. Sampling with (without)
      * replacement can be achieved by using a PoissonDistribution (BinomialDistribution)
      * randomState.
      *
      * @param input Input dataset.
      * @param subsamplingRate Fraction of the training data used for learning decision tree.
      * @param numSubsamples Number of subsamples of this RDD to take.
      * @param seed Random seed.
      * @param randomState AbstractIntegerDistribution instance used to draw random samples.
      * @return BaggedPoint dataset representation.
      */
    def convertToBaggedRDDHelper[Datum](
      input: RDD[Datum],
      subsamplingRate: Double,
      numSubsamples: Int,
      seed: Long,
      randomState: AbstractIntegerDistribution): RDD[BaggedPoint[Datum]] = {

      input.mapPartitionsWithIndex { (partitionIndex, instances) =>
        // Use random seed = seed + partitionIndex + 1 to make generation reproducible.
        // Use a different seed for each partition to ensure each partition has an independent set
        // of random numbers.
        randomState.reseedRandomGenerator(seed + partitionIndex + 1)
        instances.map {
          instance => new BaggedPoint(instance, subsamplingRate, numSubsamples, randomState)
        }
      }
    }

    if (withReplacement) {
      val poisson = new PoissonDistribution(subsamplingRate)
      convertToBaggedRDDHelper(input, subsamplingRate, numSubsamples, seed, poisson)
    } else {
      if (numSubsamples == 1 && subsamplingRate == 1.0) {
        input.map(datum => BaggedPoint(datum, Array(1)))  // Create bagged RDD without sampling
      } else {
        val binomial = new BinomialDistribution(1, subsamplingRate)
        convertToBaggedRDDHelper(input, subsamplingRate, numSubsamples, seed, binomial)
      }
    }
  }

  /**
    * Flattens a baggedRDD to a pair RDD. The key is the subsampleIndex. The instances are
    * duplicated in the flattened representation according to their specified subsampleWeight.
    * (e.g. subsampleWeight == 0 then no instances, subsampleWeight == 1 then one instance,
    * subsampleWeight == 2 then two instances, etc.)
    *
    * @param baggedRdd BaggedPoint dataset representation.
    * @return Pair RDD that contains a flattened representation of the data.
    */
  def flattenBaggedRDD[Datum](baggedRdd: RDD[BaggedPoint[Datum]], seed: Long): RDD[(Int, Datum)] = {

    val rnd = new Random(seed)

    /**
      * Converts a subsampleWeight of type Double to an Int. This is done by rounding
      * probabilistically. For example, an input subsampleWeight of 1.3 has a 70% chance of being
      * rounded to 1 and a 30% chance of being rounded to 2.
      *
      * @param subsampleWeight Weight of this instance in each subsampled dataset.
      * @return The rounded subsampleWeight.
      */
    def roundWeight(subsampleWeight: Double): Int = {
      val subsampleWeightBase = subsampleWeight.toInt
      val subsampleWeightDiff = subsampleWeight - subsampleWeightBase
      if (subsampleWeightDiff == 0)
        subsampleWeightBase
      else
        subsampleWeightBase + (if (rnd.nextFloat() < subsampleWeightDiff) 1 else 0)
    }

    baggedRdd.flatMap {
      baggedPoint => baggedPoint.subsampleWeights.zipWithIndex.flatMap {
        case (subsampleWeight, subsampleId) => {
          Array.fill(roundWeight(subsampleWeight))((subsampleId, baggedPoint.datum))
        }
      }
    }
  }
}
