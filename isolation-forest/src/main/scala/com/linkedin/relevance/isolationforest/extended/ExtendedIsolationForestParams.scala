package com.linkedin.relevance.isolationforest.extended

import org.apache.spark.ml.param.{IntParam, ParamValidators}
import com.linkedin.relevance.isolationforest.core.IsolationForestParamsBase

/**
 * Params specific to the extended isolation forest, on top of the base isolation forest params.
 */
trait ExtendedIsolationForestParams extends IsolationForestParamsBase {

  /**
   * The extension level used by the extended isolation forest.
   *  - 0 => standard iForest (random hyperplane has exactly 1 non-zero coordinate)
   *  - 1 => partially extended
   *  - 2 => fully extended in 3D (for example).
   *
   * Generally, extensionLevel + 1 = number of coordinates in the normal vector that are non-zero.
   * Must be in [0, dimensionOfSubspace].
   */
  final val extensionLevel = new IntParam(
    this,
    "extensionLevel",
    "Extension level for the random hyperplane. extensionLevel+1 = number of non-zero coordinates." +
    " 0 => standard iForest splits, dimensionOfSubspace-1 => fully extended splits",
    ParamValidators.gtEq(0)
  )
  def setExtensionLevel(value: Int): this.type = set(extensionLevel, value)
  final def getExtensionLevel: Int = $(extensionLevel)

  setDefault(
    extensionLevel -> (Int.MaxValue - 1),  // Default to fully extended (the -1 is important to avoid overflow).
  )
}
