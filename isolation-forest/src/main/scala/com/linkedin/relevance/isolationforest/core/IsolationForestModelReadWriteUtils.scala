package com.linkedin.relevance.isolationforest.core

import com.linkedin.relevance.isolationforest.IsolationTree
import com.linkedin.relevance.isolationforest.Nodes.{ExternalNode, InternalNode, Node}
import com.linkedin.relevance.isolationforest.core.NodesBase.NodeBase
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.MLWriter
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods.{compact, parse, render}

/**
 * A base trait housing code common to both standard (IsolationForest) and extended
 * isolation forest read/write logic.
 */
private[isolationforest] case object IsolationForestModelReadWriteUtils extends Logging {

  val NullNodeId: Int = -1
  val NullNumInstances: Long = -1

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //    Shared Metadata Helper
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /**
   * Stores the isolation forest model metadata.
   *
   * @param className The name of the saved class.
   * @param uid The immutable unique ID for the model.
   * @param timestamp The time when the model was saved.
   * @param sparkVersion The Spark version used to save the model.
   * @param params The model paramMap, as a `JValue`.
   * @param metadata All metadata, including the other fields not in paramMap.
   * @param metadataJson Full metadata file String (for debugging).
   */
  case class Metadata(
    className: String,
    uid: String,
    timestamp: Long,
    sparkVersion: String,
    params: JValue,
    metadata: JValue,
    metadataJson: String) {

    /**
     * Extract Params from metadata, and set them in the model instance. This works if all Params
     * implement [[org.apache.spark.ml.param.Param.jsonDecode()]].
     *
     * @param instance The model instance.
     */
    def setParams(instance: Params): Unit = {

      params match {
        case JObject(pairs) =>
          pairs.foreach { case (paramName, jsonValue) =>
            val param = instance.getParam(paramName)
            val value = param.jsonDecode(compact(render(jsonValue)))
            instance.set(param, value)
          }
        case _ =>
          throw new IllegalArgumentException(s"Cannot recognize JSON metadata: ${metadataJson}.")
      }
    }
  }

  /**
   * Read a metadata JSON file from 'metadata/' subdir. The `expectedClassName` can be empty or
   * a required class name to enforce a check.
   *
   * @param path              The top-level directory path for the model
   * @param spark             The SparkSession
   * @param expectedClassName If non-empty, we verify the loaded metadata's class matches.
   */
   def loadMetadata(
    path: String,
    spark: SparkSession,
    expectedClassName: String): Metadata = {

    val metadataPath = new Path(path, "metadata").toString
    logInfo(s"Loading model metadata from $metadataPath")

    val firstLine = spark.sparkContext.textFile(metadataPath, 1).first()
    parseMetadata(firstLine, expectedClassName)
  }

  /**
   * Parse the JSON metadata string into our Metadata container.
   */
  private def parseMetadata(
    metadataStr: String,
    expectedClassName: String): Metadata = {

    implicit val fmt = DefaultFormats
    val js = parse(metadataStr)

    val cls = (js \ "class").extract[String]
    if (expectedClassName.nonEmpty) {
      require(cls == expectedClassName,
        s"Expected class $expectedClassName, but found $cls")
    }
    val uid = (js \ "uid").extract[String]
    val ts = (js \ "timestamp").extract[Long]
    val sparkVer = (js \ "sparkVersion").extract[String]
    val p = (js \ "paramMap")

    Metadata(cls, uid, ts, sparkVer, p, js, metadataStr)
  }

  /**
   * Writes the spark.ml model metadata and Params values to disk.
   *
   * @param instance The spark.ml Model instance to save.
   * @param path The path on disk used to save the metadata.
   * @param spark The SparkSession instance to use.
   * @param extraMetadata Any extra metadata to save in addition to the model Params.
   */
  def saveMetadata(
    instance: Params,
    path: String,
    spark: SparkSession,
    extraMetadata: Option[JObject] = None): Unit = {

    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, spark, extraMetadata)
    logInfo(s"Saving IsolationForestModel metadata to path ${metadataPath}")
    spark.sparkContext.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
   * This is a helper method for [[IsolationForestModelWriter.saveMetadata()]].
   *
   * @param instance The spark.ml Model instance to save.
   * @param spark The SparkSession instance to use.
   * @param extraMetadata Any extra metadata to save in addition to the model Params.
   * @return The metadata JSON string.
   */
  private def getMetadataToSave(
                                 instance: Params,
                                 spark: SparkSession,
                                 extraMetadata: Option[JObject] = None): String = {

    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq
    val jsonParams = render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList)
    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> spark.sparkContext.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) => basicMetadata ~ jObject
      case None => basicMetadata
    }
    val metadataJson = compact(render(metadata))

    metadataJson
  }
}
