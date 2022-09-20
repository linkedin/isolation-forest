val algorithmVersion = "3.0.0"

name := s"isolation-forest_${algorithmVersion}"

organization := "com.linkedin.isolation-forest"

version := "2.0.8"

// Usage: sbt -J-DscalaVersion=2.13.8
scalaVersion := sys.props.getOrElse("scalaVersion", "2.12.16")

// Usage: sbt -J-DsparkVersion=3.1.2
val sparkVersion = sys.props.getOrElse("sparkVersion", "3.2.0")

libraryDependencies += {
  CrossVersion.partialVersion(sparkVersion) match {
    case Some((2, x)) if x < 4 =>
      "com.databricks" %% "spark-avro" % "4.0.0"

    case _ =>
      "org.apache.spark" %% "spark-avro" % sparkVersion % Provided
  }
}

libraryDependencies ++= Seq(
  "com.chuusai" %% "shapeless" % "2.3.10",
  "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
  "org.scalatestplus" %% "testng-7-5" % "3.2.13.0" % Test)

// Test
Test / run / fork := true

Test / mainClass := Some("org.testng.TestNG")

addCommandAlias(
  "testOnly",
  Seq(
    "BaggedPointTest",
    "IsolationForestModelWriteReadTest",
    "BaggedPointTest",
    "IsolationForestTest",
    "IsolationTreeTest").map { c =>
    s"test:run -testclass com.linkedin.relevance.isolationforest.${c}"
  }.mkString(" ;"))
