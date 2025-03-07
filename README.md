<img src="./assets/isolation_forest.svg" alt="Logo" width="30%"/>

# isolation-forest
[![Build Status](https://github.com/linkedin/isolation-forest/actions/workflows/ci.yml/badge.svg?branch=master&event=push)](https://github.com/linkedin/isolation-forest/actions/workflows/ci.yml?query=branch%3Amaster+event%3Apush)
[![Release](https://img.shields.io/github/v/release/linkedin/isolation-forest)](https://github.com/linkedin/isolation-forest/releases/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)

## Table of contents
- [Introduction](#introduction)
- [Features](#features)
- [Getting started](#getting-started)
  - [Building the library](#building-the-library)
  - [Add an isolation-forest dependency to your project](#add-an-isolation-forest-dependency-to-your-project)
- [Usage examples](#usage-examples)
  - [Model parameters](#model-parameters)
  - [Training and scoring](#training-and-scoring)
  - [Saving and loading a trained model](#saving-and-loading-a-trained-model)
- [ONNX conversion for portable inference](#onnx-conversion-for-portable-inference)
  - [Converting a trained model to ONNX](#converting-a-trained-model-to-onnx)
  - [Using the ONNX model for inference (example in Python)](#using-the-onnx-model-for-inference-example-in-python)
- [Performance and benchmarks](#performance-and-benchmarks)
- [Copyright and license](#copyright-and-license)
- [Contributing](#contributing)
- [References](#references)

## Introduction

This is a distributed Scala/Spark implementation of the Isolation Forest unsupervised outlier detection
algorithm. It features support for ONNX export for easy cross-platform inference. This library was created
by [James Verbus](https://www.linkedin.com/in/jamesverbus/) from the LinkedIn Anti-Abuse AI team.

## Features

* **Distributed training and scoring:** The `isolation-forest` module supports distributed training and scoring in Scala
  using Spark data structures. It inherits from the `Estimator` and `Model` classes in [Spark's ML library](https://spark.apache.org/mllib/) in
  order to take advantage of machinery such as `Pipeline`s. Model persistence on HDFS is supported.
* **Broad portability via ONNX:** The `isolation-forest-onnx` module provides Python-based converter to convert a
  trained model to ONNX format for broad portability across platforms and languages. [ONNX](https://onnx.ai/) is an open format built
  to represent machine learning models.

## Getting started

### Building the library

To build using the default of Scala 2.13.14 and Spark 3.5.1, run the following:

```bash
./gradlew build
```
This will produce a jar file in the `./isolation-forest/build/libs/` directory.

If you want to use the library with arbitrary Spark and Scala versions, you can specify this when running the
build command.

```bash
./gradlew build -PsparkVersion=3.5.1 -PscalaVersion=2.13.14
```

To force a rebuild of the library, you can use:
```bash
./gradlew clean build --no-build-cache
```

To just run the tests:
```bash
./gradlew test
```

### Add an isolation-forest dependency to your project

Please check [Maven Central](https://repo.maven.apache.org/maven2/com/linkedin/isolation-forest/) for the latest
artifact versions.

#### Gradle example

The artifacts are available in Maven Central, so you can specify the Maven Central repository in the top-level
`build.gradle` file.

```
repositories {
    mavenCentral()
}
```

Add the isolation-forest dependency to the module-level `build.gradle` file. Here is an example for a recent
spark scala version combination.

```
dependencies {
    compile 'com.linkedin.isolation-forest:isolation-forest_3.5.1_2.13:3.2.3'
}
```

#### Maven example

If you are using the Maven Central repository, declare the isolation-forest dependency in your project's `pom.xml` file.
Here is an example for a recent Spark/Scala version combination.

```
<dependency>
  <groupId>com.linkedin.isolation-forest</groupId>
  <artifactId>isolation-forest_3.5.1_2.13</artifactId>
  <version>3.2.3</version>
</dependency>
```

## Usage examples

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

Here is an example demonstrating how to import the library, create a new `IsolationForest`
instance, set the model hyperparameters, train the model, and then score the training data. `data`
is a Spark DataFrame with a column named `features` that contains a
`org.apache.spark.ml.linalg.Vector` of the attributes to use for training. In this example, the
DataFrame `data` also has a `labels` column; it is not used in the training process, but could
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

The output DataFrame, `dataWithScores`, is identical to the input `data` DataFrame but has two
additional result columns appended with their names set via model parameters; in this case, these
are named `predictedLabel` and `outlierScore`.

### Saving and loading a trained model

Once you've trained an `isolationForestModel` instance as per the instructions above, you can use the
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

## ONNX conversion for portable inference

### Converting a trained model to ONNX

The artifacts associated with the `isolation-forest-onnx` module are [available](https://pypi.org/project/isolation-forest-onnx/) in PyPI.

The ONNX converter can be installed using `pip`. It is recommended to use the same version of the converter as the
version of the `isolation-forest` library used to train the model.

```bash
pip install isolation-forest-onnx==3.2.7
```

You can then import and use the converter in Python.

```python
import os
from isolationforestonnx.isolation_forest_converter import IsolationForestConverter

# This is the same path used in the previous example showing how to save the model in Scala above.
path = '/user/testuser/isolationForestWriteTest'

# Get model data path
data_dir_path = path + '/data'
avro_model_file = os.listdir(data_dir_path)
model_file_path = data_dir_path + '/' + avro_model_file[0]

# Get model metadata file path
metadata_dir_path =  path + '/metadata'
metadata_file = os.listdir(path + '/metadata/')
metadata_file_path = metadata_dir_path + '/' + metadata_file[0]

# Convert the model to ONNX format (this will return the ONNX model in memory)
converter = IsolationForestConverter(model_file_path, metadata_file_path)
onnx_model = converter.convert()

# Convert and save the model in ONNX format (this will save the ONNX model to disk)
onnx_model_path = '/user/testuser/isolationForestWriteTest.onnx'
converter.convert_and_save(onnx_model_path)
```

### Using the ONNX model for inference (example in Python)

```python
import numpy as np
import onnx
from onnxruntime import InferenceSession

# `onnx_model_path` the same path used above in the convert and save operation
onnx_model_path = '/user/testuser/isolationForestWriteTest.onnx'
dataset_path = 'isolation-forest-onnx/test/resources/shuttle.csv'

# Load data
input_data = np.loadtxt(dataset_path, delimiter=',')
num_features = input_data.shape[1] - 1
last_col_index = num_features
print(f'Number of features for {dataset_name}: {num_features}')

# The last column is the label column
input_dict = {'features': np.delete(input_data, last_col_index, 1).astype(dtype=np.float32)}
actual_labels = input_data[:, last_col_index]

# Load the ONNX model from local disk and do inference
onx = onnx.load(onnx_model_path)
sess = InferenceSession(onx.SerializeToString())
res = sess.run(None, input_dict)

# Print scores
actual_outlier_scores = res[0]
print('ONNX Converter outlier scores:')
print(np.transpose(actual_outlier_scores[:num_examples_to_print])[0])
```

## Performance and benchmarks

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

Our implementation provides AUROC values that are in very good agreement with the results in the original
Liu et al. publication. There are a few very small discrepancies that are likely due to the limited
precision of the AUROC values reported in Liu et al.

## Copyright and license

Copyright 2019 LinkedIn Corporation
All Rights Reserved.

Licensed under the BSD 2-Clause License (the "License").
See [License](LICENSE) in the project root for license information.

## Contributing

If you would like to contribute to this project, please review the instructions [here](CONTRIBUTING.md). 

## References

* F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413–422.
* F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation-based anomaly detection,” ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 6, no. 1, p. 3, 2012.
* Shebuti Rayana (2016).  ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.
