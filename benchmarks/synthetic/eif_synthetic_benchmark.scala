/**
 * Generates three synthetic 2D datasets (single blob, two blobs, sinusoid), trains
 * Standard Isolation Forest and Extended Isolation Forest on each, scores the training
 * data and a meshgrid, and writes the results as CSVs for visualization.
 *
 * Mirrors the synthetic examples from the EIF paper (Hariri et al. 2021) and the
 * reference EIF.ipynb notebook.
 *
 * Output CSVs (written to a configurable output directory):
 *   {dataset}_{model}_data.csv    -- scored training data (x, y, outlierScore)
 *   {dataset}_{model}_grid.csv    -- scored meshgrid (x, y, outlierScore)
 *
 * Usage:
 *   spark-shell --jars /path/to/isolation-forest_3.5.5_2.12-X.Y.Z.jar
 *   scala> :load eif_synthetic_benchmark.scala
 *   scala> EIFSyntheticBenchmark.run(spark)
 *   scala> EIFSyntheticBenchmark.run(spark, outputDir = "/tmp/eif_synthetic_output")
 */

import com.linkedin.relevance.isolationforest.IsolationForest
import com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForest
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import java.io.{File, PrintWriter}
import scala.util.Random

object EIFSyntheticBenchmark {

  // ---------------------------------------------------------------------------
  // Data generation
  // ---------------------------------------------------------------------------

  /** Generates a single 2D Gaussian blob with one manual outlier at (3.3, 3.3).
   *  Matches EIF.ipynb: mean=[0,0], cov=I, N=500, seed=1. */
  def generateSingleBlob(seed: Int = 1): Array[(Double, Double)] = {
    val rng = new Random(seed)
    val n = 500
    val points = Array.tabulate(n) { _ =>
      (rng.nextGaussian(), rng.nextGaussian())
    }
    points(0) = (3.3, 3.3) // manual outlier
    points
  }

  /** Generates two 2D Gaussian blobs at [10,0] and [0,10].
   *  Matches EIF.ipynb: cov=I each, N=500 total (250 each), seed=1. */
  def generateTwoBlobs(seed: Int = 1): Array[(Double, Double)] = {
    val rng = new Random(seed)
    val n = 250
    val blob1 = Array.tabulate(n) { _ =>
      (10.0 + rng.nextGaussian(), rng.nextGaussian())
    }
    val blob2 = Array.tabulate(n) { _ =>
      (rng.nextGaussian(), 10.0 + rng.nextGaussian())
    }
    blob1 ++ blob2
  }

  /** Generates sinusoidal data: y = sin(x) + noise.
   *  Matches EIF.ipynb: N=1000, x in [0, 8*pi], noise ~ N(0, 0.25). */
  def generateSinusoid(seed: Int = 1): Array[(Double, Double)] = {
    val rng = new Random(seed)
    val n = 1000
    Array.tabulate(n) { _ =>
      val x = rng.nextDouble() * 8.0 * math.Pi
      val y = math.sin(x) + rng.nextGaussian() / 4.0
      (x, y)
    }
  }

  /** Generates a 2D meshgrid as an array of (x, y) points. */
  def generateMeshgrid(xMin: Double, xMax: Double, yMin: Double, yMax: Double,
                       nPoints: Int = 100): Array[(Double, Double)] = {
    val xs = (0 until nPoints).map(i => xMin + i * (xMax - xMin) / (nPoints - 1))
    val ys = (0 until nPoints).map(i => yMin + i * (yMax - yMin) / (nPoints - 1))
    for (y <- ys.toArray; x <- xs.toArray) yield (x, y)
  }

  // ---------------------------------------------------------------------------
  // DataFrame helpers
  // ---------------------------------------------------------------------------

  /** Converts an array of (x, y) points to a Spark DataFrame with a features vector. */
  def pointsToDF(spark: SparkSession, points: Array[(Double, Double)]): DataFrame = {
    import spark.implicits._
    points.map { case (x, y) => (x, y, Vectors.dense(x, y)) }
      .toSeq.toDF("x", "y", "features")
  }

  // ---------------------------------------------------------------------------
  // Model training and scoring
  // ---------------------------------------------------------------------------

  /** Trains StandardIF on the data and returns scored DataFrames for data and grid. */
  def scoreWithStandardIF(data: DataFrame, grid: DataFrame, numTrees: Int, seed: Int): (DataFrame, DataFrame) = {
    val model = new IsolationForest()
      .setNumEstimators(numTrees)
      .setMaxSamples(256.0)
      .setContamination(0.1)
      .setRandomSeed(seed)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
    val fitted = model.fit(data)
    (fitted.transform(data), fitted.transform(grid))
  }

  /** Trains ExtendedIF at the given extension level and returns scored DataFrames. */
  def scoreWithExtendedIF(data: DataFrame, grid: DataFrame, extensionLevel: Int,
                          numTrees: Int, seed: Int): (DataFrame, DataFrame) = {
    val model = new ExtendedIsolationForest()
      .setNumEstimators(numTrees)
      .setMaxSamples(256.0)
      .setContamination(0.1)
      .setRandomSeed(seed)
      .setExtensionLevel(extensionLevel)
      .setFeaturesCol("features")
      .setPredictionCol("predictedLabel")
      .setScoreCol("outlierScore")
    val fitted = model.fit(data)
    (fitted.transform(data), fitted.transform(grid))
  }

  // ---------------------------------------------------------------------------
  // CSV output
  // ---------------------------------------------------------------------------

  /** Writes a scored DataFrame (with x, y, outlierScore columns) to a single-partition CSV. */
  def writeCSV(scored: DataFrame, path: String): Unit = {
    val rows = scored.select("x", "y", "outlierScore").collect()
    val pw = new PrintWriter(new File(path))
    try {
      pw.println("x,y,outlierScore")
      rows.foreach { row =>
        pw.println(f"${row.getDouble(0)}%.6f,${row.getDouble(1)}%.6f,${row.getDouble(2)}%.6f")
      }
    } finally pw.close()
    println(s"  Wrote ${rows.length} rows to $path")
  }

  // ---------------------------------------------------------------------------
  // Per-dataset runner
  // ---------------------------------------------------------------------------

  case class SyntheticDataset(
    name: String,
    points: Array[(Double, Double)],
    gridXMin: Double, gridXMax: Double,
    gridYMin: Double, gridYMax: Double,
    numTrees: Int
  )

  def runDataset(spark: SparkSession, ds: SyntheticDataset, outputDir: String, seed: Int): Unit = {
    println(s"\n=== ${ds.name} (${ds.points.length} points, ${ds.numTrees} trees) ===")

    val dataDF = pointsToDF(spark, ds.points).cache()
    val gridDF = pointsToDF(spark, generateMeshgrid(ds.gridXMin, ds.gridXMax, ds.gridYMin, ds.gridYMax)).cache()

    // Standard IF (equivalent to extension level 0)
    println("  -> Standard IF")
    val (scoredDataStd, scoredGridStd) = scoreWithStandardIF(dataDF, gridDF, ds.numTrees, seed)
    writeCSV(scoredDataStd, s"$outputDir/${ds.name}_standard_if_data.csv")
    writeCSV(scoredGridStd, s"$outputDir/${ds.name}_standard_if_grid.csv")

    // Extended IF with extension level = 1 (fully extended for 2D data)
    println("  -> Extended IF (ext=1)")
    val (scoredDataExt, scoredGridExt) = scoreWithExtendedIF(dataDF, gridDF, 1, ds.numTrees, seed)
    writeCSV(scoredDataExt, s"$outputDir/${ds.name}_extended_if_data.csv")
    writeCSV(scoredGridExt, s"$outputDir/${ds.name}_extended_if_grid.csv")

    dataDF.unpersist()
    gridDF.unpersist()
  }

  // ---------------------------------------------------------------------------
  // Main entry point
  // ---------------------------------------------------------------------------

  def run(spark: SparkSession, outputDir: String = "eif_synthetic_output", seed: Int = 1): Unit = {
    new File(outputDir).mkdirs()

    val datasets = Seq(
      SyntheticDataset("single_blob", generateSingleBlob(seed),
        gridXMin = -5.0, gridXMax = 5.0, gridYMin = -5.0, gridYMax = 5.0, numTrees = 200),
      SyntheticDataset("two_blobs", generateTwoBlobs(seed),
        gridXMin = -5.0, gridXMax = 15.0, gridYMin = -5.0, gridYMax = 15.0, numTrees = 500),
      SyntheticDataset("sinusoid", generateSinusoid(seed),
        gridXMin = -5.0, gridXMax = 30.0, gridYMin = -3.0, gridYMax = 3.0, numTrees = 500)
    )

    datasets.foreach(ds => runDataset(spark, ds, outputDir, seed))

    println(s"\nDone. CSVs written to $outputDir/")
  }
}
