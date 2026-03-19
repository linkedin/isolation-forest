# isolation-forest-onnx

A converter for the LinkedIn Spark/Scala [isolation forest](https://github.com/linkedin/isolation-forest) model format to [ONNX](https://onnx.ai/) format for broad portability across platforms and languages.

**Note:** ONNX conversion is supported for the standard `IsolationForestModel` only. The `ExtendedIsolationForestModel` uses hyperplane-based splits that are not compatible with the axis-aligned tree ensemble representation used by the ONNX converter.

## Installation

```bash
pip install isolation-forest-onnx
```

It is recommended to use the same version of the converter as the version of the `isolation-forest` library used to train the model.

## Converting a trained model to ONNX

```python
import os
from isolationforestonnx.isolation_forest_converter import IsolationForestConverter

# Path where the trained IsolationForestModel was saved in Scala
path = '/user/testuser/isolationForestWriteTest'

# Get model data path
data_dir_path = path + '/data'
avro_model_file = os.listdir(data_dir_path)
model_file_path = data_dir_path + '/' + avro_model_file[0]

# Get model metadata file path
metadata_dir_path = path + '/metadata'
metadata_file = os.listdir(metadata_dir_path)
metadata_file_path = metadata_dir_path + '/' + metadata_file[0]

# Convert the model to ONNX format (returns the ONNX model in memory)
converter = IsolationForestConverter(model_file_path, metadata_file_path)
onnx_model = converter.convert()

# Convert and save the model in ONNX format
onnx_model_path = '/user/testuser/isolationForestWriteTest.onnx'
converter.convert_and_save(onnx_model_path)
```

## Using the ONNX model for inference

```python
import numpy as np
import onnx
from onnxruntime import InferenceSession

onnx_model_path = '/user/testuser/isolationForestWriteTest.onnx'
dataset_path = 'shuttle.csv'

# Load data
input_data = np.loadtxt(dataset_path, delimiter=',')
num_features = input_data.shape[1] - 1
last_col_index = num_features

# The last column is the label column
input_dict = {'features': np.delete(input_data, last_col_index, 1).astype(dtype=np.float32)}

# Load the ONNX model and run inference
onx = onnx.load(onnx_model_path)
sess = InferenceSession(onx.SerializeToString())
res = sess.run(None, input_dict)

# Print scores
outlier_scores = res[0]
print(np.transpose(outlier_scores[:10])[0])
```

## License

BSD 2-Clause License. See [LICENSE](https://github.com/linkedin/isolation-forest/blob/master/LICENSE) for details.
