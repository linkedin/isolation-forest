# isolation-forest
[![Build Status](https://travis-ci.org/linkedin/isolation-forest.svg?branch=master)](https://travis-ci.org/linkedin/isolation-forest)
[![Download](https://api.bintray.com/packages/linkedin/maven/isolation-forest/images/download.svg)](https://bintray.com/linkedin/maven/isolation-forest/_latestVersion)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)

This is a Scala/Spark implementation of the Isolation Forest unsupervised outlier detection
algorithm. This library was created by [James Verbus](https://www.linkedin.com/in/jamesverbus/) from
the LinkedIn Anti-Abuse AI team.

This library supports distributed training and scoring using Spark data structures. It inherits from
the ``Estimator`` and ``Model`` classes in [Spark's ML library](https://spark.apache.org/docs/2.3.0/ml-guide.html)
in order to take advantage of machinery such as ``Pipeline``s. Model persistence on HDFS is
supported.

## Copyright

Copyright 2019 LinkedIn Corporation
All Rights Reserved.

Licensed under the BSD 2-Clause License (the "License").
See [License](LICENSE) in the project root for license information.

## How to use

### Building the library

It is recommended to use Scala 2.11.8 and Spark 2.3.0. To build, run the following:

```bash
./gradlew build
```
This will produce a jar file in the ``./isolation-forest/build/libs/`` directory.

If you want to use the library with Spark 2.4 (and the Scala 2.11.8 default), you can specify this when running the
build command.

```bash
./gradlew build -PsparkVersion=2.4.3
```

You can also build an artifact with Spark 2.4 (or 3.0) and Scala 2.12.

```bash
./gradlew build -PsparkVersion=3.0.0 -PscalaVersion=2.12.11
```

### Add an isolation-forest dependency to your project

Please check [Bintray](https://bintray.com/beta/#/linkedin/maven/isolation-forest) for the latest
artifact versions.

#### Gradle example

The artifacts are available in JCenter, so you can specify the JCenter repository in the top-level build.gradle file.

```
repositories {
    jcenter()
}
```

Add the isolation-forest dependency to the module-level build.gradle file. Here are some examples for multiple recent
Spark/Scala version combinations.

```
dependencies {
    compile 'com.linkedin.isolation-forest:isolation-forest_2.3.0_2.11:1.0.1'
}
```
```
dependencies {
    compile 'com.linkedin.isolation-forest:isolation-forest_2.4.3_2.11:1.0.1'
}
```
```
dependencies {
    compile 'com.linkedin.isolation-forest:isolation-forest_2.4.3_2.12:1.0.1'
}
```
```
dependencies {
    compile 'com.linkedin.isolation-forest:isolation-forest_3.0.0_2.12:1.0.1'
}
```

#### Maven example

If you are using the Maven Central repository, declare the isolation-forest dependency in your project's pom.xml file.
Here are some examples for multiple recent Spark/Scala version combinations.

```
<dependency>
  <groupId>com.linkedin.isolation-forest</groupId>
  <artifactId>isolation-forest_2.3.0_2.11</artifactId>
  <version>1.0.1</version>
</dependency>
```
```
<dependency>
  <groupId>com.linkedin.isolation-forest</groupId>
  <artifactId>isolation-forest_2.4.3_2.11</artifactId>
  <version>1.0.1</version>
</dependency>
```
```
<dependency>
  <groupId>com.linkedin.isolation-forest</groupId>
  <artifactId>isolation-forest_2.4.3_2.12</artifactId>
  <version>1.0.1</version>
</dependency>
```
```
<dependency>
  <groupId>com.linkedin.isolation-forest</groupId>
  <artifactId>isolation-forest_3.0.0_2.12</artifactId>
  <version>1.0.1</version>
</dependency>
```

### Model parameters

| Parameter          | Default Value    | Description                                                                                                                                                                                                                                                                                                                                                                          |
|--------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| numEstimators      | 100              | The number of trees in the ensemble.                                                                                                                                                                                                                                                                                                                                                 |
| maxSamples         | 256              | The number of samples used to train each tree. If this value is between 0.0 and 1.0, then it is treated as a fraction. If it is >1.0, then it is treated as a count.                                                                                                                                                                                                                 |
| contamination      | 0.0              | The fraction of outliers in the training data set. If this is set to 0.0, it speeds up the training and all predicted labels will be false. The model and outlier scores are otherwise unaffected by this parameter.                                                                                                                                                                 |
| contaminationError | 0.0              | The error allowed when calculating the threshold required to achieve the specified contamination fraction. The default is 0.0, which forces an exact calculation of the threshold. The exact calculation is slow and can fail for large datasets. If there are issues with the exact calculation, a good choice for this parameter is often 1% of the specified contamination value. |
| maxFeatures        | 1.0              | The number of features used to train each tree. If this value is between 0.0 and 1.0, then it is treated as a fraction. If it is >1.0, then it is treated as a count.                                                                                                                                                                                                                |
| bootstrap          | false            | If true, draw sample for each tree with replacement. If false, do not sample with replacement.                                                                                                                                                                                                                                                                                       |
| randomSeed         | 1                | The seed used for the random number generator.                                                                                                                                                                                                                                                                                                                                       |
| featuresCol        | "features"       | The feature vector. This column must exist in the input DataFrame for training and scoring.                                                                                                                                                                                                                                                                                          |
| predictionCol      | "predictedLabel" | The predicted label. This column is appended to the input DataFrame upon scoring.                                                                                                                                                                                                                                                                                                    |
| scoreCol           | "outlierScore"   | The outlier score. This column is appended to the input DataFrame upon scoring.                                                                                                                                                                                                                                                                                                      |

### Training and scoring

Here is an example demonstrating how to import the library, create a new ``IsolationForest``
instance, set the model hyperparameters, train the model, and then score the training data.``data``
is a Spark DataFrame with a column named ``features`` that contains a 
``org.apache.spark.ml.linalg.Vector`` of the attributes to use for training. In this example, the
DataFrame ``data`` also has a ``labels`` column; it is not used in the training process, but could
be useful for model evaluation.

```scala
import com.linkedin.relevance.isolationforest._
import org.apache.spark.ml.feature.VectorAssembler

/**
  * Load and prepare data
  */

// Dataset from http://odds.cs.stonybrook.edu/shuttle-dataset/
val rawData = spark.read
  .format("csv")
  .option("comment", "#")
  .option("header", "false")
  .option("inferSchema", "true")
  .load("isolation-forest/src/test/resources/shuttle.csv")

val cols = rawData.columns
val labelCol = cols.last
 
val assembler = new VectorAssembler()
  .setInputCols(cols.slice(0, cols.length - 1))
  .setOutputCol("features")
val data = assembler
  .transform(rawData)
  .select(col("features"), col(labelCol).as("label"))

// scala> data.printSchema
// root
//  |-- features: vector (nullable = true)
//  |-- label: integer (nullable = true)

/**
  * Train the model
  */

val contamination = 0.1
val isolationForest = new IsolationForest()
  .setNumEstimators(100)
  .setBootstrap(false)
  .setMaxSamples(256)
  .setMaxFeatures(1.0)
  .setFeaturesCol("features")
  .setPredictionCol("predictedLabel")
  .setScoreCol("outlierScore")
  .setContamination(contamination)
  .setContaminationError(0.01 * contamination)
  .setRandomSeed(1)

val isolationForestModel = isolationForest.fit(data)
 
/**
  * Score the training data
  */

val dataWithScores = isolationForestModel.transform(data)

// scala> dataWithScores.printSchema
// root
//  |-- features: vector (nullable = true)
//  |-- label: integer (nullable = true)
//  |-- outlierScore: double (nullable = false)
//  |-- predictedLabel: double (nullable = false)
```

The output DataFrame, ``dataWithScores``, is identical to the input ``data`` DataFrame but has two
additional result columns appended with their names set via model parameters; in this case, these
are named ``predictedLabel`` and ``outlierScore``.

### Saving and loading a trained model

Once you've trained an ``isolationForestModel`` instance as per the instructions above, you can use the
following commands to save the model to HDFS and reload it as needed.

```scala
val path = "/user/testuser/isolationForestWriteTest"

/**
  * Persist the trained model on disk
  */

// You can ensure you don't overwrite an existing model by removing .overwrite from this command
isolationForestModel.write.overwrite.save(path)

/**
  * Load the saved model from disk
  */

val isolationForestModel2 = IsolationForestModel.load(path)
```

## Validation

The original 2008 "Isolation forest" paper by Liu et al. published the AUROC results obtained by
applying the algorithm to 12 benchmark outlier detection datasets. We applied our implementation of
the isolation forest algorithm to the same 12 datasets using the same model parameter values used in
the original paper. We used 10 trials per dataset each with a unique random seed and averaged the
result. The quoted uncertainty is the one-sigma error on the mean.

| Dataset                                                                            | Expected mean AUROC (from Liu et al.) | Observed mean AUROC (from this implementation) |
|------------------------------------------------------------------------------------|---------------------------------------|------------------------------------------------|
| [Http (KDDCUP99)](http://odds.cs.stonybrook.edu/http-kddcup99-dataset/)            | 1.00                                  | 0.99973 &plusmn; 0.00007                       |
| [ForestCover](http://odds.cs.stonybrook.edu/forestcovercovertype-dataset/)         | 0.88                                  | 0.903 &plusmn; 0.005                           |
| [Mulcross](https://www.openml.org/d/40897)                                         | 0.97                                  | 0.9926 &plusmn; 0.0006                         |
| [Smtp (KDDCUP99)](http://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/)            | 0.88                                  | 0.907 &plusmn; 0.001                           |
| [Shuttle](http://odds.cs.stonybrook.edu/shuttle-dataset/)                          | 1.00                                  | 0.9974 &plusmn; 0.0014                         |
| [Mammography](http://odds.cs.stonybrook.edu/mammography-dataset/)                  | 0.86                                  | 0.8636 &plusmn; 0.0015                         |
| [Annthyroid](http://odds.cs.stonybrook.edu/annthyroid-dataset/)                    | 0.82                                  | 0.815 &plusmn; 0.006                           |
| [Satellite](http://odds.cs.stonybrook.edu/satellite-dataset/)                      | 0.71                                  | 0.709 &plusmn; 0.004                           |
| [Pima](http://odds.cs.stonybrook.edu/pima-indians-diabetes-dataset/)               | 0.67                                  | 0.651 &plusmn; 0.003                           |
| [Breastw](http://odds.cs.stonybrook.edu/breast-cancer-wisconsin-original-dataset/) | 0.99                                  | 0.9862 &plusmn; 0.0003                         |
| [Arrhythmia](http://odds.cs.stonybrook.edu/arrhythmia-dataset/)                    | 0.80                                  | 0.804 &plusmn; 0.002                           |
| [Ionosphere](http://odds.cs.stonybrook.edu/ionosphere-dataset/)                    | 0.85                                  | 0.8481 &plusmn; 0.0002                         |

Our implementation provides AUROC values that are in very good agreement the results in the original
Liu et al. publication. There are a few very small discrepancies that are likely due the limited
precision of the AUROC values reported in Liu et al.

## Contributions

If you would like to contribute to this project, please review the instructions [here](CONTRIBUTING.md).

## References

* F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413–422.
* F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation-based anomaly detection,” ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 6, no. 1, p. 3, 2012.
* Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.
