# 将基于TensorFlow GraphDef的模型导入TensorFlow.js
基于TensorFlow GraphDef的模型（通常通过Python API创建）可以采用以下格式之一保存：

* TensorFlow SavedModel
* Frozen Model
* Session Bundle
* Tensorflow Hub module

所有上述格式都可以通过TensorFlow.js转换器转换为TensorFlow.js Web友好格式，可以直接加载到TensorFlow.js进行推理。

（`注意`：TensorFlow已弃用会话包格式，请将模型迁移到SavedModel格式。）

# 要求
转换过程需要Python环境; 你可能想用pipenv或virtualenv的其中一个。要安装转换器，请运行以下命令：
```js
pip install tensorflowjs
```
将TensorFlow模型导入TensorFlow.js需要两个步骤。首先，将现有模型转换为TensorFlow.js Web格式，然后将其加载到TensorFlow.js中。

# 步骤1.将现有TensorFlow模型转换为TensorFlow.js Web格式
运行pip包提供的转换器脚本：

用法：

* 转换SavedModel示例：
```sh
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```
* 转换Frozen Model示例：
```sh
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

* 转换Session Bundle示例：
```sh
tensorflowjs_converter \
    --input_format=tf_session_bundle \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/session_bundle \
    /mobilenet/web_model
```
* 转换Tensorflow Hub module示例：
```sh
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```
|位置参数 |	描述 |
|---|---|
|input_path |	保存的模型目录，会话包目录，冻结模型文件或TensorFlow Hub模块句柄或路径的完整路径。|
|output_path|	所有输出工件的路径。|

|选项|	描述|
|---|---|
|--input_format|	输入模型的格式，使用tf_saved_model表示SavedModel，tf_frozen_model表示Frozen Model，tf_session_bundle表示Session Bundle，tf_hub表示TensorFlow Hubmodule，keras表示Keras HDF5。|
|--output_node_names|	输出节点的名称，以逗号分隔。|
|--saved_model_tags|	仅适用于SavedModel转换，以逗号分隔格式加载MetaGraphDef的标签。默认为serve。|
|--signature_name|	仅适用于TensorFlow Hub模块转换，签名加载。默认为default。请参阅https://www.tensorflow.org/hub/common_signatures/。|

使用以下命令获取详细帮助消息：
```sh
tensorflowjs_converter --help
```

# 转换器生成的文件

上面的转换脚本生成3种类型的文件：

* web_model.pb （数据流图）
* weights_manifest.json （重量清单文件）
* group1-shard\*of\* （二进制权重文件的集合）

例如，以下是在以下位置转换并提供的MobileNet模型：
```
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/optimized_model.pb
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/weights_manifest.json
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard1of5
...
https://storage.cloud.google.com/tfjs-models/savedmodel/mobilenet_v1_1.0_224/group1-shard5of5
```
## 第2步：在浏览器中加载并运行

安装tfjs-converter npm包
```sh
yarn add @tensorflow/tfjs 要么 npm install @tensorflow/tfjs
``

实例化FrozenModel类并运行推理。
```js
import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'https://.../mobilenet/web_model.pb';
const WEIGHTS_URL = 'https://.../mobilenet/weights_manifest.json';

const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
const cat = document.getElementById('cat');
model.execute({input: tf.fromPixels(cat)});
```
查看我们的[MobileNet演示版](https://github.com/tensorflow/tfjs-converter/demo/mobilenet/README.md)。

如果您的服务器请求访问模型文件的凭据，您可以提供可选的RequestOption参数，该参数将直接传递给fetch函数调用。
```js
const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL,
    {credentials: 'include'});
```
有关详细信息，请参阅[fetch()文档](https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch)。

# 支持的操作
目前TensorFlow.js仅支持一组有限的TensorFlow Ops。查看 [完整列表](https://github.com/tensorflow/tfjs-converter/docs/supported_ops.md)。如果您的模型使用任何不受支持的操作，则tensorflowjs_converter脚本将失败并生成模型中不受支持的操作的列表。请提交[问题](https://github.com/tensorflow/tfjs/issues)，告诉我们您需要支持的操作。

# 仅加载重量
如果您只想加载权重，可以使用以下代码段。
```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```