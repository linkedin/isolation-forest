package com.linkedin.relevance.isolationforest

import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, LongParam, Param, ParamValidators, Params}


/**
  * Params for the isolation forest model.
  */
trait IsolationForestParams extends Params {

  final val numEstimators = new IntParam(this, "numEstimators", "The number of trees in the" +
    " ensemble.", ParamValidators.gt(0.0))
  def setNumEstimators(value: Int): this.type = set(numEstimators, value)
  final def getNumEstimators: Int = $(numEstimators)

  final val maxSamples = new DoubleParam(this, "maxSamples", "The number of samples used to" +
    " train each tree. If this value is between 0.0 and 1.0, then it is treated as a fraction. If" +
    " it is >1.0, then it is treated as a count.", ParamValidators.gt(0.0))
  def setMaxSamples(value: Double): this.type = set(maxSamples, value)
  final def getMaxSamples: Double = $(maxSamples)

  final val contamination = new DoubleParam(this, "contamination", "The fraction of outliers in" +
    " the training data set. If this is set to 0.0, it speeds up the training and all predicted" +
    " labels will be false. The model and outlier scores are otherwise unaffected by this parameter.",
    ParamValidators.inRange(0.0, 0.5, lowerInclusive = true, upperInclusive = false))
  def setContamination(value: Double): this.type = set(contamination, value)
  final def getContamination: Double = $(contamination)

  final val maxFeatures = new DoubleParam(this, "maxFeatures", "The number of features used to" +
    " train each tree. If this value is between 0.0 and 1.0, then it is treated as a fraction." +
    " If it is >1.0, then it is treated as a count.", ParamValidators.gt(0.0))
  def setMaxFeatures(value: Double): this.type = set(maxFeatures, value)
  final def getMaxFeatures: Double = $(maxFeatures)

  final val bootstrap = new BooleanParam(this, "bootstrap", "If true, draw sample for each tree" +
    " with replacement. If false, do not sample with replacement.")
  def setBootstrap(value: Boolean): this.type = set(bootstrap, value)
  final def getBootstrap: Boolean = $(bootstrap)

  final val contaminationError = new DoubleParam(this, "contaminationError", "The error" +
    " allowed when calculating the threshold required to achieve the specified contamination" +
    " fraction. The default is 0.0, which forces an exact calculation of the threshold. The" +
    " exact calculation is slow and can fail for large datasets. If there are issues with the" +
    " exact calculation, a good choice for this parameter is often 1% of the specified" +
    " contamination value.",
    ParamValidators.inRange(0.0, 1, lowerInclusive = true, upperInclusive = true))
  def setContaminationError(value: Double): this.type = set(contaminationError, value)
  final def getContaminationError: Double = $(contaminationError)

  final val randomSeed = new LongParam(this, "randomSeed", "The seed used for the random" +
    " number generator.",  ParamValidators.gt(0.0))
  def setRandomSeed(value: Long): this.type = set(randomSeed, value)
  final def getRandomSeed: Long = $(randomSeed)

  final val featuresCol = new Param[String](this, "featuresCol", "The feature vector.")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  final def getFeaturesCol: String = $(featuresCol)

  final val predictionCol = new Param[String](this, "predictionCol", "The predicted label.")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  final def getPredictionCol: String = $(predictionCol)

  final val scoreCol = new Param[String](this, "scoreCol", "The outlier score.")
  def setScoreCol(value: String): this.type = set(scoreCol, value)
  final def getScoreCol: String = $(scoreCol)

  setDefault(
    numEstimators -> 100,
    maxSamples -> 256,
    contamination -> 0.0,
    contaminationError -> 0.0,
    maxFeatures -> 1.0,
    bootstrap -> false,
    randomSeed -> 1,
    featuresCol -> "features",
    predictionCol -> "predictedLabel",
    scoreCol -> "outlierScore")
}
