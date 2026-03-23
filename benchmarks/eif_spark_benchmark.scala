/**
 * Canonical Spark benchmark script for validating the Extended Isolation Forest (EIF) algorithm
 * against standard benchmark datasets.
 *
 * Compares three models:
 *   1. Standard Isolation Forest (IsolationForest)
 *   2. Extended Isolation Forest (ExtendedIsolationForest) with extensionLevel = 0
 *   3. Extended Isolation Forest (ExtendedIsolationForest) with extensionLevel = dim - 1 (max)
 *
 * Models 1 and 2 should produce statistically equivalent results (validating that EIF at ext=0
 * matches standard IF). Model 3 tests the fully extended algorithm.
 *
 * Runs across 13 benchmark datasets (12 from Liu et al. 2008 + cardio from the EIF paper),
 * computing AUROC and AUPRC over multiple random seeds using sklearn-compatible metrics
 * (trapezoidal ROC, step-function PR).
 *
 * Results are validated against two reference sources on the SAME dataset files:
 *   - linkedin/isolation-forest README: Published AUROC +/- SEM for StandardIF.
 *   - Reference Python EIF (sahandha/eif): Run on our exact files to validate Spark EIF_max.
 *
 * Note: the EIF paper (Hariri et al. 2021) Table 3 values are NOT directly comparable because
 * the paper used differently preprocessed versions of the benchmark datasets. The reference
 * Python implementation run on our files confirms this -- it matches our Spark results, not the
 * paper's Table 3.
 *
 * Usage:
 *   spark-shell --jars /path/to/isolation-forest_3.5.5_2.12-X.Y.Z.jar
 *   scala> :load eif_spark_benchmark.scala
 *   scala> EIFBenchmark.run(spark)                          // default: 100 trees, 10 iterations
 *   scala> EIFBenchmark.run(spark, numIter = 1)             // quick smoke test
 *   scala> EIFBenchmark.run(spark, saveModelDir = Some("/tmp/eif_models"))  // save fitted models
 */

import com.linkedin.relevance.isolationforest.IsolationForest
import com.linkedin.relevance.isolationforest.extended.ExtendedIsolationForest
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import java.io.{File, PrintWriter}

object EIFBenchmark {

  /** Writes the toString of the first tree's root node to a text file. */
  def saveTreeStructure(path: String, treeString: String): Unit = {
    val pw = new PrintWriter(new File(path))
    try pw.print(treeString)
    finally pw.close()
    println(s"    Wrote tree structure to $path")
  }

  /**
   * Container for the benchmark results of a single model on a single dataset.
   *
   * @param dataset        The dataset filename (e.g. "http.csv").
   * @param dimension      The number of features in the dataset.
   * @param contamination  The fraction of outliers assumed in the dataset.
   * @param model          The model identifier ("StandardIF", "ExtendedIF_0", or "ExtendedIF_max").
   * @param extensionLevel The extension level used (0 for standard/ext0, dim-1 for max).
   * @param meanAuroc      Mean AUROC across all iterations.
   * @param semAuroc       Standard error of the mean AUROC.
   * @param stdAuroc       Standard deviation of AUROC across iterations.
   * @param meanAuprc      Mean AUPRC across all iterations.
   * @param semAuprc       Standard error of the mean AUPRC.
   * @param stdAuprc       Standard deviation of AUPRC across iterations.
   */
  case class BenchmarkResult(
    dataset: String,
    dimension: Int,
    contamination: Double,
    model: String,
    extensionLevel: Int,
    meanAuroc: Double,
    semAuroc: Double,
    stdAuroc: Double,
    meanAuprc: Double,
    semAuprc: Double,
    stdAuprc: Double
  )

  /**
   * Loads a benchmark CSV dataset. Assumes the CSV has no header, all columns but the last are
   * numeric features, and the last column is a binary label (0 = normal, 1 = outlier).
   * Lines starting with '#' are treated as comments and skipped.
   *
   * @param spark The SparkSession.
   * @param path  Path to the CSV file (local or HDFS).
   * @return A DataFrame with columns "features" (Vector) and "label".
   */
  def loadData(spark: SparkSession, path: String): DataFrame = {
    val rawData = spark.read
      .format("csv")
      .option("comment", "#")
      .option("header", "false")
      .option("inferSchema", "true")
      .load(path)

    val cols = rawData.columns
    val featureCols = cols.slice(0, cols.length - 1)
    val labelCol = cols.last

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    assembler
      .transform(rawData)
      .select(col("features"), col(labelCol).as("label"))
  }

  /**
   * Computes AUROC and AUPRC from a scored DataFrame using sklearn-compatible methods.
   *
   * AUROC is computed via the trapezoidal rule on the ROC curve (same as sklearn's roc_auc_score).
   * AUPRC is computed via step-function interpolation (same as sklearn's average_precision_score):
   *   AP = sum (Recall_n - Recall_{n-1}) * Precision_n
   *
   * This differs from Spark's BinaryClassificationMetrics.areaUnderPR(), which uses trapezoidal
   * interpolation in PR space -- known to give incorrect results on imbalanced datasets (Davis &
   * Goadrich 2006).
   *
   * Ties in scores are grouped so that all samples with the same score are processed together,
   * matching sklearn's behavior.
   *
   * @param scored The DataFrame output from model.transform(), containing outlierScore and label.
   * @return A tuple of (AUROC, AUPRC).
   */
  def getMetrics(scored: DataFrame): (Double, Double) = {
    val scoreAndLabel = scored.select(
      col("outlierScore").cast("double"),
      col("label").cast("double")
    ).collect().map(row => (row.getDouble(0), row.getDouble(1)))

    // Sort by score descending (higher score = more anomalous)
    val sorted = scoreAndLabel.sortBy(-_._1)
    val n = sorted.length
    val totalPos = sorted.count(_._2 == 1.0).toDouble
    val totalNeg = n - totalPos

    if (totalPos == 0 || totalNeg == 0) return (Double.NaN, Double.NaN)

    var tp = 0.0
    var fp = 0.0
    var prevTpr = 0.0
    var prevFpr = 0.0
    var prevRecall = 0.0
    var auroc = 0.0
    var auprc = 0.0

    var i = 0
    while (i < n) {
      val currentScore = sorted(i)._1
      // Process all samples with the same score together (handle ties like sklearn)
      while (i < n && sorted(i)._1 == currentScore) {
        if (sorted(i)._2 == 1.0) tp += 1.0 else fp += 1.0
        i += 1
      }

      val tpr = tp / totalPos
      val fpr = fp / totalNeg
      val precision = tp / (tp + fp)
      val recall = tpr

      // AUROC: trapezoidal rule (matches sklearn's roc_auc_score)
      auroc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0

      // AUPRC: step-function interpolation (matches sklearn's average_precision_score)
      auprc += (recall - prevRecall) * precision

      prevTpr = tpr
      prevFpr = fpr
      prevRecall = recall
    }

    (auroc, auprc)
  }

  /**
   * Evaluates the standard IsolationForest over multiple random seeds.
   * Optionally saves the fitted model from the first seed to disk.
   *
   * @param data          The input DataFrame with "features" and "label" columns.
   * @param contamination The assumed outlier fraction in the dataset.
   * @param numTrees      Number of isolation trees in the ensemble.
   * @param numIter       Number of iterations (random seeds 1..numIter).
   * @param saveDir       If Some(path), saves the seed=1 fitted model to path/standard_if/.
   * @return A tuple of (meanAUROC, semAUROC, stdAUROC, meanAUPRC, semAUPRC, stdAUPRC).
   */
  def evaluateStandardIF(data: DataFrame, contamination: Double, numTrees: Int, numIter: Int, saveDir: Option[String] = None): (Double, Double, Double, Double, Double, Double) = {
    val results = (1 to numIter).map { seed =>
      println(s"    [StandardIF] Iteration $seed/$numIter ...")
      val model = new IsolationForest()
        .setNumEstimators(numTrees)
        .setMaxSamples(256.0)
        .setContamination(contamination)
        .setRandomSeed(seed)
        .setFeaturesCol("features")
        .setPredictionCol("predictedLabel")
        .setScoreCol("outlierScore")
      val fittedModel = model.fit(data)
      if (seed == 1) saveDir.foreach { dir =>
        val path = s"$dir/standard_if"
        println(s"    Saving StandardIF model to $path")
        fittedModel.write.overwrite().save(path)
        saveTreeStructure(s"$path/expectedTreeStructure.txt", fittedModel.isolationTrees.head.node.toString)
      }
      val scored = fittedModel.transform(data)
      getMetrics(scored)
    }
    computeStats(results, numIter)
  }

  /**
   * Evaluates the ExtendedIsolationForest at a given extension level over multiple random seeds.
   * Optionally saves the fitted model from the first seed to disk.
   *
   * @param data           The input DataFrame with "features" and "label" columns.
   * @param contamination  The assumed outlier fraction in the dataset.
   * @param extensionLevel The EIF extension level (0 = axis-parallel like standard IF,
   *                       dim-1 = fully extended).
   * @param numTrees       Number of isolation trees in the ensemble.
   * @param numIter        Number of iterations (random seeds 1..numIter).
   * @param saveDir        If Some(path), saves the seed=1 fitted model to
   *                       path/extended_if_ext{level}/.
   * @return A tuple of (meanAUROC, semAUROC, stdAUROC, meanAUPRC, semAUPRC, stdAUPRC).
   */
  def evaluateExtendedIF(data: DataFrame, contamination: Double, extensionLevel: Int, numTrees: Int, numIter: Int, saveDir: Option[String] = None): (Double, Double, Double, Double, Double, Double) = {
    val results = (1 to numIter).map { seed =>
      println(s"    [ExtendedIF ext=$extensionLevel] Iteration $seed/$numIter ...")
      val model = new ExtendedIsolationForest()
        .setNumEstimators(numTrees)
        .setMaxSamples(256.0)
        .setContamination(contamination)
        .setRandomSeed(seed)
        .setExtensionLevel(extensionLevel)
        .setFeaturesCol("features")
        .setPredictionCol("predictedLabel")
        .setScoreCol("outlierScore")
      val fittedModel = model.fit(data)
      if (seed == 1) saveDir.foreach { dir =>
        val path = s"$dir/extended_if_ext$extensionLevel"
        println(s"    Saving ExtendedIF (ext=$extensionLevel) model to $path")
        fittedModel.write.overwrite().save(path)
        saveTreeStructure(s"$path/expectedTreeStructure.txt", fittedModel.extendedIsolationTrees.head.extendedNode.toString)
      }
      val scored = fittedModel.transform(data)
      getMetrics(scored)
    }
    computeStats(results, numIter)
  }

  /**
   * Computes summary statistics (mean, standard deviation, standard error of the mean) for
   * AUROC and AUPRC from a sequence of per-iteration results. Uses Bessel's correction (ddof=1)
   * for the standard deviation. Returns NaN for std/sem when numIter < 2.
   *
   * @param results  Per-iteration (AUROC, AUPRC) tuples.
   * @param numIter  Number of iterations.
   * @return A tuple of (meanAUROC, semAUROC, stdAUROC, meanAUPRC, semAUPRC, stdAUPRC).
   */
  def computeStats(results: IndexedSeq[(Double, Double)], numIter: Int): (Double, Double, Double, Double, Double, Double) = {
    val aurocs = results.map(_._1)
    val auprcs = results.map(_._2)

    val meanAuroc = aurocs.sum / numIter
    val meanAuprc = auprcs.sum / numIter

    val stdAuroc = if (numIter > 1) math.sqrt(aurocs.map(x => math.pow(x - meanAuroc, 2)).sum / (numIter - 1)) else Double.NaN
    val stdAuprc = if (numIter > 1) math.sqrt(auprcs.map(x => math.pow(x - meanAuprc, 2)).sum / (numIter - 1)) else Double.NaN

    val semAuroc = stdAuroc / math.sqrt(numIter)
    val semAuprc = stdAuprc / math.sqrt(numIter)

    (meanAuroc, semAuroc, stdAuroc, meanAuprc, semAuprc, stdAuprc)
  }

  /**
   * Runs all three models (StandardIF, ExtendedIF_0, ExtendedIF_max) on a single dataset
   * and returns the benchmark results. Caches the input data for the duration of evaluation.
   *
   * @param spark         The SparkSession.
   * @param path          Path to the benchmark CSV file.
   * @param contamination The assumed outlier fraction for this dataset.
   * @param numTrees      Number of isolation trees per model.
   * @param numIter       Number of random seed iterations per model.
   * @param saveModelDir  If Some(path), saves seed=1 fitted models under path/{dataset}/.
   * @return A Seq of three BenchmarkResult entries (one per model variant).
   */
  def compareModelsOnDataset(spark: SparkSession, path: String, contamination: Double, numTrees: Int, numIter: Int, saveModelDir: Option[String] = None): Seq[BenchmarkResult] = {
    val data = loadData(spark, path).cache()
    val dimension = data.select("features").head().getAs[org.apache.spark.ml.linalg.Vector](0).size
    val datasetName = path.split("/").last.replace(".csv", "")
    val maxExtLevel = math.max(0, dimension - 1)

    // Per-dataset save directory
    val datasetSaveDir = saveModelDir.map(dir => s"$dir/$datasetName")

    println(s"\n=== Evaluating: $datasetName (dim=$dimension, contamination=$contamination) ===")

    // 1. Standard Isolation Forest (IsolationForest class)
    println("  -> Standard IF (IsolationForest)")
    val (mAucStd, sAucStd, sdAucStd, mAprStd, sAprStd, sdAprStd) =
      evaluateStandardIF(data, contamination, numTrees, numIter, datasetSaveDir)

    // 2. Extended Isolation Forest with extensionLevel = 0 (should match Standard IF)
    println("  -> Extended IF (ext=0)")
    val (mAucExt0, sAucExt0, sdAucExt0, mAprExt0, sAprExt0, sdAprExt0) =
      evaluateExtendedIF(data, contamination, 0, numTrees, numIter, datasetSaveDir)

    // 3. Extended Isolation Forest with extensionLevel = dim - 1 (fully extended)
    println(s"  -> Extended IF (ext=$maxExtLevel)")
    val (mAucExtMax, sAucExtMax, sdAucExtMax, mAprExtMax, sAprExtMax, sdAprExtMax) =
      evaluateExtendedIF(data, contamination, maxExtLevel, numTrees, numIter, datasetSaveDir)

    data.unpersist()

    Seq(
      BenchmarkResult(datasetName, dimension, contamination, "StandardIF", 0, mAucStd, sAucStd, sdAucStd, mAprStd, sAprStd, sdAprStd),
      BenchmarkResult(datasetName, dimension, contamination, "ExtendedIF_0", 0, mAucExt0, sAucExt0, sdAucExt0, mAprExt0, sAprExt0, sdAprExt0),
      BenchmarkResult(datasetName, dimension, contamination, "ExtendedIF_max", maxExtLevel, mAucExtMax, sAucExtMax, sdAucExtMax, mAprExtMax, sAprExtMax, sdAprExtMax)
    )
  }

  /** Expected AUROC from the original Liu et al. (2008) isolation forest paper.
   *  These are rounded point estimates reported in the paper (no uncertainty available).
   *  Shown for informational context only -- not used for validation. */
  val liuResults: Map[String, Double] = Map(
    "http"        -> 1.00,
    "cover"       -> 0.88,
    "mulcross"    -> 0.97,
    "smtp"        -> 0.88,
    "shuttle"     -> 1.00,
    "mammography" -> 0.86,
    "annthyroid"  -> 0.82,
    "satellite"   -> 0.71,
    "pima"        -> 0.67,
    "breastw"     -> 0.99,
    "arrhythmia"  -> 0.80,
    "ionosphere"  -> 0.85
  )

  /** Published benchmark AUROC from the linkedin/isolation-forest README.
   *  Maps dataset name (without .csv) to (observed_mean_AUROC, SEM_AUROC).
   *  The observed values are the mean over 10 trials reported in the library's documentation. */
  val linkedinIFResults: Map[String, (Double, Double)] = Map(
    "http"        -> (0.99973, 0.00007),
    "cover"       -> (0.903,   0.005),
    "mulcross"    -> (0.9926,  0.0006),
    "smtp"        -> (0.907,   0.001),
    "shuttle"     -> (0.9974,  0.0014),
    "mammography" -> (0.8636,  0.0015),
    "annthyroid"  -> (0.815,   0.006),
    "satellite"   -> (0.709,   0.004),
    "pima"        -> (0.651,   0.003),
    "breastw"     -> (0.9862,  0.0003),
    "arrhythmia"  -> (0.804,   0.002),
    "ionosphere"  -> (0.8481,  0.0002)
  )

  /** Reference Python EIF (sahandha/eif eif_old.py) run on our exact dataset files.
   *  100 trees, 256 samples, 10 iterations.
   *  Maps dataset name to (ext0_AUROC, ext0_AUROC_SEM, ext0_AUPRC, ext0_AUPRC_SEM,
   *                         max_AUROC, max_AUROC_SEM, max_AUPRC, max_AUPRC_SEM).
   *  Used to validate that our Spark implementation matches the reference on the same data. */
  case class RefResult(auroc: Double, aurocSem: Double, auprc: Double, auprcSem: Double)
  case class RefPair(ext0: RefResult, max: RefResult)
  val refPythonResults: Map[String, RefPair] = Map(
    "http"        -> RefPair(RefResult(0.9939, 0.0001, 0.3791, 0.0043), RefResult(0.9939, 0.0003, 0.3709, 0.0093)),
    "cover"       -> RefPair(RefResult(0.8724, 0.0100, 0.0487, 0.0040), RefResult(0.6617, 0.0094, 0.0129, 0.0004)),
    "mulcross"    -> RefPair(RefResult(0.9595, 0.0025, 0.5381, 0.0173), RefResult(0.9405, 0.0048, 0.4457, 0.0217)),
    "smtp"        -> RefPair(RefResult(0.8968, 0.0024, 0.0041, 0.0001), RefResult(0.8570, 0.0025, 0.0143, 0.0031)),
    "shuttle"     -> RefPair(RefResult(0.9975, 0.0001, 0.9805, 0.0010), RefResult(0.9932, 0.0002, 0.8179, 0.0028)),
    "mammography" -> RefPair(RefResult(0.8676, 0.0023, 0.2293, 0.0130), RefResult(0.8639, 0.0016, 0.1837, 0.0043)),
    "annthyroid"  -> RefPair(RefResult(0.8217, 0.0037, 0.3143, 0.0070), RefResult(0.6505, 0.0033, 0.1828, 0.0045)),
    "satellite"   -> RefPair(RefResult(0.6999, 0.0043, 0.6641, 0.0056), RefResult(0.7396, 0.0052, 0.7108, 0.0051)),
    "pima"        -> RefPair(RefResult(0.6753, 0.0048, 0.5136, 0.0047), RefResult(0.6403, 0.0041, 0.4933, 0.0038)),
    "breastw"     -> RefPair(RefResult(0.9873, 0.0005, 0.9704, 0.0016), RefResult(0.9841, 0.0006, 0.9593, 0.0021)),
    "arrhythmia"  -> RefPair(RefResult(0.7960, 0.0035, 0.4624, 0.0049), RefResult(0.8026, 0.0033, 0.4896, 0.0035)),
    "ionosphere"  -> RefPair(RefResult(0.8556, 0.0016, 0.8078, 0.0021), RefResult(0.9061, 0.0014, 0.8764, 0.0022)),
    "cardio"      -> RefPair(RefResult(0.9175, 0.0034, 0.5463, 0.0128), RefResult(0.9308, 0.0024, 0.5470, 0.0089))
  )

  /** Display sort order: StandardIF first, then ExtendedIF_0, then ExtendedIF_max. */
  val modelOrder: Map[String, Int] = Map("StandardIF" -> 0, "ExtendedIF_0" -> 1, "ExtendedIF_max" -> 2)

  /**
   * Prints the benchmark results as a Markdown table. For each dataset, rows are ordered:
   * StandardIF, ExtendedIF_0, ExtendedIF_max.
   *
   * Reference columns:
   *   - Liu AUROC: Expected values from Liu et al. (2008), for informational context.
   *   - LI IF AUROC +/- SEM: linkedin/isolation-forest published AUROC (same datasets, same library).
   *   - Ref Python AUROC/AUPRC +/- SEM: reference EIF (sahandha/eif) run on our exact files.
   *     Only shown for ExtendedIF_0 and ExtendedIF_max rows (the reference is EIF, not standard IF).
   *
   * @param results The collected BenchmarkResult entries from all datasets.
   */
  def printResultsTable(results: Seq[BenchmarkResult]): Unit = {
    val sorted = results.sortBy(r => (r.dataset, modelOrder.getOrElse(r.model, 99)))

    println("\n## Benchmark Results\n")
    println("| Dataset | Dim | Ext | Model | AUROC (mean +/- SEM) | AUPRC (mean +/- SEM) | Liu AUROC | LI IF AUROC | Ref Python AUROC | Ref Python AUPRC |")
    println("|---------|-----|-----|-------|----------------------|----------------------|-----------|-------------|------------------|------------------|")
    for (r <- sorted) {
      // Liu et al. (2008) -- informational
      val liuAurocStr = liuResults.get(r.dataset) match {
        case Some(v) => f"$v%.2f"
        case None => "-"
      }

      // linkedin/isolation-forest published results -- has (mean, SEM)
      val liIfAurocStr = linkedinIFResults.get(r.dataset) match {
        case Some((mean, sem)) => f"$mean%.4f+/-$sem%.4f"
        case None => "-"
      }

      // Reference Python EIF results -- pick ext0 or max depending on model
      val (refPyAurocStr, refPyAuprcStr) = refPythonResults.get(r.dataset) match {
        case Some(pair) =>
          val ref = r.model match {
            case "ExtendedIF_0" => pair.ext0
            case "ExtendedIF_max" => pair.max
            case _ => null
          }
          if (ref != null) (f"${ref.auroc}%.4f+/-${ref.aurocSem}%.4f", f"${ref.auprc}%.4f+/-${ref.auprcSem}%.4f")
          else ("-", "-")
        case None => ("-", "-")
      }

      println(f"| ${r.dataset}%-15s | ${r.dimension}%3d | ${r.extensionLevel}%3d | ${r.model}%-14s | ${r.meanAuroc}%.4f +/- ${r.semAuroc}%.4f | ${r.meanAuprc}%.4f +/- ${r.semAuprc}%.4f | $liuAurocStr%-9s | $liIfAurocStr%-11s | $refPyAurocStr%-16s | $refPyAuprcStr%-16s |")
    }
    println()
  }

  /**
   * Main entry point. Runs the full benchmark suite across all 13 datasets.
   *
   * @param spark        The SparkSession (provided automatically by spark-shell).
   * @param numTrees     Number of isolation trees per model (default 100).
   * @param numIter      Number of random seed iterations per model (default 10).
   *                     Use numIter >= 2 for meaningful SEM and p-value calculations.
   * @param saveModelDir If Some(path), saves the seed=1 fitted models to disk, organized as
   *                     path/{dataset}/{standard_if,extended_if_ext0,extended_if_extN}/.
   */
  def run(spark: SparkSession, numTrees: Int = 100, numIter: Int = 10, saveModelDir: Option[String] = None): Unit = {
    // Resolve dataset path relative to the script's location (works when :load-ed from spark-shell)
    val dataPath = {
      val scriptDir = new java.io.File(".").getCanonicalPath
      s"$scriptDir/datasets"
    }

    val datasets = Seq(
      (s"$dataPath/http.csv",        0.004),
      (s"$dataPath/cover.csv",       0.009),
      (s"$dataPath/mulcross.csv",    0.10),
      (s"$dataPath/smtp.csv",        0.003),
      (s"$dataPath/shuttle.csv",     0.07),
      (s"$dataPath/mammography.csv", 0.0232),
      (s"$dataPath/annthyroid.csv",  0.0742),
      (s"$dataPath/satellite.csv",   0.32),
      (s"$dataPath/pima.csv",        0.35),
      (s"$dataPath/breastw.csv",     0.35),
      (s"$dataPath/arrhythmia.csv",  0.15),
      (s"$dataPath/ionosphere.csv",  0.36),
      (s"$dataPath/cardio.csv",      0.096)
    )

    val allResults = datasets.flatMap { case (path, cont) =>
      compareModelsOnDataset(spark, path, cont, numTrees, numIter, saveModelDir)
    }

    printResultsTable(allResults)
  }
}
