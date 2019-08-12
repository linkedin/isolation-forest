package com.linkedin.relevance.isolationforest

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Dataset, Encoders, SparkSession}


object TestUtils {

  case class LabeledDataPointVector(features: Vector, label: Double)
  case class MammographyRecord(feature0: Double, feature1: Double, feature2: Double, feature3: Double,
                               feature4: Double, feature5: Double, label: Double)
  case class ScoringResult(features: Vector, label: Double, predictedLabel: Double, outlierScore: Double)
  case class ShuttleRecord(feature0: Double, feature1: Double, feature2: Double, feature3: Double,
                           feature4: Double, feature5: Double, feature6: Double, feature7: Double,
                           feature8: Double, label: Double)


  def getSparkSession: SparkSession = {

    val sparkConf: SparkConf = {
      // Turn on Kryo serialization by default
      val conf = new SparkConf()
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      conf.set("spark.driver.host", "localhost")
      conf
    }

    // local context with 4 threads
    SparkSession.builder
      .master("local[4]")
      .appName("testing-spark")
      .config(sparkConf)
      .getOrCreate()
  }

  def loadMammographyData(session: SparkSession): Dataset[LabeledDataPointVector] = {

    import session.implicits._

    val mammographyRecordSchema = Encoders.product[MammographyRecord].schema

    // Open source dataset from http://odds.cs.stonybrook.edu/mammography-dataset/
    val rawData = session.read
      .format("csv")
      .option("comment", "#")
      .option("header", "false")
      .schema(mammographyRecordSchema)
      .load("src/test/resources/mammography.csv")
      .as[MammographyRecord]

    val assembler = new VectorAssembler()
      .setInputCols(Array("feature0", "feature1", "feature2", "feature3", "feature4", "feature5"))
      .setOutputCol("features")

    val data = assembler
      .transform(rawData)
      .select("features", "label")
      .as[LabeledDataPointVector]

    data
  }

  def loadShuttleData(session: SparkSession): Dataset[LabeledDataPointVector] = {

    import session.implicits._

    val shuttleRecordSchema = Encoders.product[ShuttleRecord].schema

    // Open source dataset from http://odds.cs.stonybrook.edu/shuttle-dataset/
    val rawData = session.read
      .format("csv")
      .option("comment", "#")
      .option("header", "false")
      .schema(shuttleRecordSchema)
      .load("src/test/resources/shuttle.csv")
      .as[ShuttleRecord]

    val assembler = new VectorAssembler()
      .setInputCols(Array("feature0", "feature1", "feature2", "feature3", "feature4", "feature5",
        "feature6", "feature7", "feature8"))
      .setOutputCol("features")

    val data = assembler
      .transform(rawData)
      .select("features", "label")
      .as[LabeledDataPointVector]

    data
  }

  def readCsv(path: String) : Array[Array[Float]] = {

    val bufferedSource = scala.io.Source.fromFile(path)
    val data = bufferedSource
      .getLines()
      .filter(x => x(0) != '#')  // Remove comments
      .map(_.split(",").map(_.trim.toFloat))
      .toArray
    bufferedSource.close()

    data
  }
}
