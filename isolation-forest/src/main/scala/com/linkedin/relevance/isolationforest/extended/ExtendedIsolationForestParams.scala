package com.linkedin.relevance.isolationforest.extended

import org.apache.spark.ml.param.{IntParam, ParamValidators}
import com.linkedin.relevance.isolationforest.core.IsolationForestParamsBase

/**
 * Params specific to the extended isolation forest, on top of the base isolation forest params.
 */
trait ExtendedIsolationForestParams extends IsolationForestParamsBase {

  /**
   * The extension level used by the extended isolation forest. Valid range is [0, numFeatures - 1]
   * where numFeatures is the resolved feature subspace dimensionality.
   *   - 0 => standard iForest (random hyperplane has exactly 1 non-zero coordinate)
   *   - numFeatures - 1 => fully extended (all coordinates may be non-zero)
   *
   * If not set, defaults to fully extended (numFeatures - 1) at fit time.
   */
  final val extensionLevel = new IntParam(
    this,
    "extensionLevel",
    "Extension level for the random hyperplane. extensionLevel+1 = number of non-zero coordinates." +
      " 0 => standard iForest splits, numFeatures-1 => fully extended splits." +
      " If not set, defaults to fully extended at fit time.",
    ParamValidators.gtEq(0),
  )
  def setExtensionLevel(value: Int): this.type = set(extensionLevel, value)
  final def getExtensionLevel: Int = $(extensionLevel)
}
