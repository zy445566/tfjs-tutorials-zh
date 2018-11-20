# 保存并加载tf.Model
本教程介绍如何在TensorFlow.js中保存和加载模型。保存和加载模型是一项重要的功能。例如，如何保存仅由浏览器中可用的数据（例如，来自附加传感器的图像和音频数据）微调的模型的权重，以便在用户加载时模型将处于已调整的状态页面又来了？还要考虑Layers API允许您tf.Model在浏览器中从头开始创建模型这一个问题。你如何保存以这种方式创建的模型？自版本0.11.1起，TensorFlow.js中提供的save / load API解决了这些问题。

`注意`：本文档是关于保存和加载tf.Models（即tfjs-layers API中的Keras样式模型）。tf.FrozenModel目前还不支持保存和加载（即从TensoFlow SavedModels 加载的模型），并且正在积极开展工作。

# 保存tf.Model
让我们从最基本，最轻松的方式开始，将tf.Model：保存到Web浏览器的本地存储。本地存储是标准的客户端数据存储。保存在那里的数据可以在同一页面的多个负载中持续存在。您可以在此MDN页面上了解有关它的更多信息 。

假设你有一个tf.Model名为的对象model。无论是从头开始使用Layers API还是从预训练的Keras模型加载/微调，您都可以使用一行代码将其保存到本地存储：
```js
const saveResult = await model.save('localstorage://my-model-1');
```
有些事情值得指出：

* 这种保存方法采用类似于URL的字符串参数，该参数以方案开头。在这种情况下，我们使用该localstorage://指定将模型保存到本地存储。
* 你需要记住这个路径。在保存到本地存储的情况下，路径只是一个任意字符串，用于唯一标识要保存的模型。例如，当您从本地存储加载模型时，将使用它。
* 这种保存方法是异步的，因此您需要使用then或者await如果其完成形成其他操作的前提条件。
* model.save的返回值是一个JSON对象，它携带一些可能有用的信息，例如模型拓扑和权重的字节大小。
* 任何tf.Model，无论它是否由 tf.sequential 构成，它包含哪些类型的层，都可以这种方式保存。
下表列出了所有当前支持的保存模型目的地及其相应的方案和示例。

|保存方法 |	方案协议 |	代码示例 |
|---|---|---|
|本地存储（浏览器）|	localstorage://|	await model.save('localstorage://my-model-1');|
|IndexedDB（浏览器）|	indexeddb://|	await model.save('indexeddb://my-model-1');|
|触发文件下载（浏览器）|	downloads:// |	await model.save('downloads://my-model-1');|
|HTTP请求（浏览器）|	http:// 或 https://|	await model.save|('http://model-server.domain/upload');|
|使用文件系统（Node.js）|	file://|	await model.save('file:///tmp/my-model-1');|

我们将在以下部分中扩展一些保存路线。

## IndexedDB的
IndexedDB 是大多数主流Web浏览器支持的另一个客户端数据存储。与本地存储不同，它更好地支持存储大型二进制数据（BLOB）和更大的配额。因此，tf.Model与本地存储相比，保存到IndexedDB通常可以提供更好的存储效率和更大的大小限制。

## 文件下载
该方案后面的字符串是将要下载的文件名称的前缀。例如，该行将 model.save('downloads://my-model-1')导致浏览器下载两个共享相同文件名前缀的文件：

1. 一个名为的文本JSON文件my-model-1.json，它在其modelTopology字段中包含模型的拓扑，并在其字段中显示权重清单 weightsManifest。
2. 一个二进制文件，带有名为的权重值my-model-1.weights.bin。
这些文件的格式与tensorflowjs转换器从Keras HDF5文件转换的工件格式相同。

`注意`：某些浏览器要求用户在同时下载多个文件之前授予权限。

## HTTP请求
如果tf.Model.save使用HTTP / HTTPS URL调用，则模型的拓扑和权重将通过POST请求发送到指定的HTTP服务器 。POST请求的主体具有一个名为的格式 multipart/form-data。它是用于将文件上载到服务器的标准MIME格式。正文由两个文件组成，文件名model.json和文件名 model.weights.bin。文件格式与downloads://方案触发的下载文件格式相同（参见上文）。此 文档字符串 包含一个Python代码片段，演示了如何使用烧瓶 Web框架以及Keras和TensorFlow来处理源自save请求的有效负载并将其重新构建为服务器内存中的Keras Model对象。

通常，您的HTTP服务器对请求有特殊约束和要求，例如HTTP方法，标头和身份验证凭据。您可以save通过将URL字符串参数替换为调用来获得对请求的这些方面的细粒度控制tf.io.browserHTTPRequest。它是一个更详细的API，但它在控制由此产生的HTTP请求时提供了更大的灵活性save。例如：
```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```
## 使用文件系统
TensorFlow.js可以在Node.js中使用。有关更多详细信息，请参阅 tfjs-node项目。与Web浏览器不同，Node.js可以直接访问本地文件系统。因此，您可以将tf.Models 保存到文件系统，这与在Keras中将模型保存到磁盘的方式非常相似。为此，首先确保已导入@tensorflow/tfjs-node包，例如，使用Node.js的require语法：
```js
require('@tensorflow/tfjs-node');
```
导入后，file://URL 该方案可用于模型保存和加载。对于模型保存，该方案后面是要保存模型工件的目录的路径，例如：
```js
await model.save('file:///tmp/my-model-1');
```
上面的命令将在目录中生成一个model.json文件和一个weights.bin文件/tmp/my-model-1。这两个文件的格式与上面的“文件下载”和“HTTP请求”部分中描述的文件格式相同。保存模型后，可以将其加载回运行TensorFlow.js的Node.js程序，或者为TensorFlow.js的浏览器版本提供服务。要实现前者，请tf.loadModel()使用model.json文件路径调用：
```js
const model = await tf.loadModel('file:///tmp/my-model-1/model.json');
```
要实现后者，请将保存的文件作为Web服务器的静态文件提供。

# 加载tf.Model
tf.Model如果之后无法加载模型，则保存s的功能将无用。通过tf.loadModel使用基于方案的类似URL的字符串参数调用来完成模型加载。tf.Model.save在大多数情况下，字符串参数是对称的 。下表给出了支持的加载路径的摘要：

|载入方案 |	方案协议 |	代码示例 |
|---|---|---|
|本地存储（浏览器）|	localstorage://|	await tf.loadModel('localstorage://my-model-1');|
|IndexedDB（浏览器）|	indexeddb://|	await tf.loadModel('indexeddb://my-model-1');|
|用户上传的文件（浏览器）|	N / A|	await tf.loadModel(tf.io.browserFiles([modelJSONFile, weightsFile]));|
|HTTP请求（浏览器）|	http:// 或 https://|	await tf.loadModel('http://model-server.domain/download/model.json');|
|使用文件系统（Node.js）|	file://|	await tf.loadModel('file:///tmp/my-model-1/model.json');|
在所有加载路由中， 如果加载成功则tf.loadModel返回一个（Promiseof）tf.Model对象，如果失败则抛出一个错误。

从本地存储或IndexedDB加载与保存完全对称。但是，从用户上传的文件加载与从浏览器下载文件完全不对称。特别是，用户上传的文件不表示为类似URL的字符串。相反，他们被指定为Array的 文件对象。典型的工作流程是让用户使用HTML 文件输入 元素从本地文件系统中选择文件
```html
<input name="json-upload" type="file" />
<input name="weights-upload" type="file" />
```
这些将在浏览器中显示为两个“选择文件”按钮，用户可以使用这些按钮来选择文件。一旦用户分别在两个文件输入中选择了model.json文件和权重文件，文件对象将在相应的HTML元素下可用，并且它们可用于加载tf.Model 如下：
```js
const jsonUpload = document.getElementById('json-upload');
const weightsUpload = document.getElementById('weights-upload');

const model = await tf.loadModel(
    tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
```
从HTTP请求加载模型对于通过HTTP请求保存模式也略微不对称。特别是，tf.loadModel将URL或路径作为model.json文件，如上表中的示例所示。这是自TensorFlow.js初始发布以来一直存在的API。

管理存储在浏览器Local Storage和IndexedDB中的模型
正如你在上面学到，你可以存储tf.Model的拓扑结构和权在用户的客户端浏览器的数据存储，包括本地存储和索引资料库，通过使用代码，如 model.save('localstorage://my-model')和model.save('indexeddb://my-model')。但到目前为止，您如何找到存储的模型？这可以通过使用tf.ioAPI 附带的模型管理方法来实现 ：
```js
// List models in Local Storage.
console.log(await tf.io.listModels());
```
方法的返回值listModels不仅包括存储模型的路径，还包括一些关于它们的简短元数据，例如拓扑和权重的字节大小。

管理API还允许您复制，移动或删除现有模型。例如：
```js
// Copy model from existing path to a new path.
// Copying between Local Storage and IndexedDB is supported.
tf.io.copyModel('localstorage://my-model', 'indexeddb://cloned-model');

// Move model from a path to another.
// Moving between Local Storage and IndexedDB is supported.
tf.io.moveModel('localstorage://my-model', 'indexeddb://cloned-model');

// Remove model.
tf.io.removeModel('indexeddb://cloned-model');
```
# 将保存的tf.Model转换为Keras格式
如上所述，有两种方法可以保存tf.Model as文件：

* 通过文件从Web浏览器下载，使用方案downloads://
* 使用该file:// 方案将模型直接写入Node.js中的本机文件系统 。使用tensorflowjs转换器，您可以将这些文件转换为HDF5格式，然后可以将其加载到Python中的Keras中。例如：
```js
# Suppose you have downloaded `my-model-1.json`, accompanied by a weights file.

pip install tensorflowjs

tensorflowjs_converter \
    --input_format tensorflowjs --output_format keras \
    ./my-model-1.json /tmp/my-model-1.h5
```