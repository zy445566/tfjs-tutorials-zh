# 将Keras模型导入TensorFlow.js
Keras模型（通常通过Python API创建）可以以多种格式之一保存。“whole model”格式可以转换为TensorFlow.js图层格式，可以直接加载到TensorFlow.js进行推理或进一步培训。

目标TensorFlow.js图层格式是一个包含model.json文件和一组二进制格式的分片权重文件的目录。该model.json文件包含模型拓扑（又名 "architecture" 或 "graph"：层的描述及其连接方式）和权重文件的清单。

# 要求
转换过程需要Python环境; 你可能想用pipenv或virtualenv的其中一个。首先要安装转换器，请使用`pip install tensorflowjs`。

将Keras模型导入TensorFlow.js分为两步。首先，将现有的Keras模型转换为TF.js图层格式，然后将其加载到TensorFlow.js中。

## 步骤1.将现有的Keras模型转换为TF.js图层格式
Keras模型通常通过model.save(filepath)，生成一个包含模型拓扑和权重的HDF5（.h5）文件。要将此类文件转换为TF.js图层格式，请运行以下命令，其中path/to/my_model.h5是源Keras.h5文件，并且path/to/tfjs_target_dir是TF.js文件的目标输出目录：
```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
### 或者：使用Python API直接导出到TF.js图层格式
如果您在Python中使用Keras模型，则可以将其直接导出为TensorFlow.js图层格式，如下所示：
```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## 第2步：将模型加载到TensorFlow.js中
使用Web服务器为您在步骤1中生成的转换后的模型文件提供服务。请注意，您可能需要将服务器配置为允许跨源资源共享（CORS），以便允许使用JavaScript获取文件。

然后通过提供model.json文件的URL将模型加载到TensorFlow.js中：
```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadModel('https://foo.bar/tfjs_artifacts/model.json');
```
现在，该模型已准备好进行推理，评估或重新培训。例如，加载的模型可以立即用于进行预测：
```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```
许多[TensorFlow.js示例](https://github.com/tensorflow/tfjs-examples)采用此方法，使用已在Google云端存储上转换和托管的预训练模型。

`请注意`，您使用model.json文件名引用整个模型。 loadModel(...)获取model.json，然后发出额外的HTTP（S）请求以获取权model.json重清单中引用的分片权重文件。这种方法允许所有这些文件由浏览器缓存（也可能由互联网上的其他缓存服务器缓存），因为model.json和权重分片都小于典型的缓存文件大小限制。因此，模型很可能在随后的场合加载更快。

# 支持的功能
TensorFlow.js图层目前仅支持使用标准Keras构造的Keras模型。使用不受支持的操作或图层的模型（例如自定义图层，Lambda图层，自定义损耗函数或自定义指标）无法自动导入，因为它们依赖于无法可靠地转换为JavaScript的Python代码。