# 转移学习 - 训练神经网络以预测网络摄像头数据
在开始之前，我们强烈建议您使用该演示。 [试试吧！](https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html)

在[核心概念教程](./core-concepts.md)中，我们学习了如何使用Tensors和Ops来执行基本线性代数。

在[卷积图像分类器](./mnist.md)教程中，我们学习了如何构建卷积图像分类器以识别来自MNIST数据集的手写数字。

在[将Keras模型导入TensorFlow.js](./import-keras.md)教程中，我们学习了如何将预训练的Keras模型移植到浏览器中进行推理。

在本教程中，我们将使用转移学习从网络摄像头数据（姿势，对象，面部表情等）预测用户定义的类型，并通过将每个姿势分配给“向上”，“向下”，“向左”和“向右”来玩Pacman。

# 关于游戏
游戏分为三个阶段。

1. `数据收集`：播放器将来自网络摄像头的图像与上，下，左，右四个类中的每一个相关联。
2. `训练`：训练神经网络从输入图像预测类。
3. `推理/播放`：使用我们训练的模型从网络摄像头数据进行上，下，左，右预测，并将其输入Pacman游戏！

# 关于模型
要了解在合理的时间量从网络摄像头不同的类分类，我们将重新训练，或微调，一个预训练的 MobileNet 模型，使用内部激活（从MobileNet的内部层的输出）作为输入到我们的新模型。

为此，我们实际上在页面上有两个模型。

一个模型将是预先训练的MobileNet模型，该模型被截断以输出内部激活。我们称之为“截断的MobileNet模型”。加载到浏览器后，此模型未经过培训。

第二个模型将截断的MobileNet模型的内部激活的输出作为输入，并将预测4个输出类（上，下，左和右）中的每一个的概率。这是我们实际在浏览器中训练的模型。

通过使用MobileNet的内部激活，我们可以重用MobileNet已经学习的功能，通过相对较少的再训练来预测1000类ImageNet。

# 关于本教程
要在本地运行代码，您需要安装以下依赖项：

* Node.js8.9或更高版本
* Yarn或NPM CLI
这些说明使用Yarn，但是如果您熟悉NPM CLI并且更喜欢使用它，那么它仍然可以使用。

您可以通过克隆repo并构建演示来运行示例的代码：
```sh
git clone https://github.com/tensorflow/tfjs-examples
cd tfjs-examples/webcam-transfer-learning
yarn
yarn watch
```
上面的[tfjs-examples/webcam-transfer-learning](https://github.com/tensorflow/tfjs-examples/tree/master/webcam-transfer-learning)目录是完全独立的，因此您可以将其复制以启动您自己的项目。

`注意`：这种方法与机器学习采用的方法不同 。可训练的机器使用K-最近邻（KNN）对来自预训练的SqueezeNet模型的预测进行分类，而这种方法使用从MobileNet的内部激活训练的第二神经网络。KNN图像分类器在较少量的数据下工作得更好，但具有传递学习的神经网络更好地概括。可以两个演示一起玩，探索两种不同的网络摄像头预测方式有何不同！

# 数据
在我们训练模型之前，我们需要一种从网络摄像头中获取Tensors的方法。

我们提供了一个webcam.js被调用的类，Webcam能从`<video>`标签中读取图像作为TensorFlow.js Tensor。

我们来看一下capture方法吧Webcam。
```js
capture() {
  return tf.tidy(() => {
    const webcamImage = tf.fromPixels(this.webcamElement);
    const croppedImage = this.cropImage(webcamImage);
    const batchedImage = croppedImage.expandDims(0);

    return batchedImage.toFloat().div(oneTwentySeven).sub(one);
  });
}
```
让我们解析这些代码。
```js
const webcamImage = tf.fromPixels(this.webcamElement);
```
此行从webcam`<video>`元素读取单个帧并返回 Tensor形状[height, width, 3]。最内层尺寸3对应于三个通道RGB。

有关支持的输入HTML元素类型，请参阅tf.fromPixels的文档。
```js
const croppedImage = this.cropImage(webcamImage);
```
设置方形网络摄像头元素时，网络摄像头源的自然宽高比为矩形（浏览器会在矩形图像周围放置空白区域以使其成为方形）。

但是，MobileNet模型需要方形输入图像。此行[224, 224]从网络摄像头元素中裁剪出一个方形中心块。请注意，有更多的代码Webcam增加了视频元素的大小，因此我们可以裁剪一个[224, 224]方块而不会获得白色填充。
```js
const batchedImage = croppedImage.expandDims(0);
```
expandDims创建一个大小为1的新外部尺寸。在这种情况下，我们从网络摄像头读取的裁剪图像具有形状[224, 224, 3]。调用 expandDims(0)将此张量重新整形为[1, 224, 224, 3]，表示一批单个图像。MobileNet期望批量输入。
```js
batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
```
在这一行中，我们将图像转换为浮点并在-1和1之间将其标准化（这就是模型的训练方式）。我们知道默认情况下图像中的值介于0到255之间，因此为了在-1和1之间进行归一化，我们除以127并减去1。
```js
return tf.tidy(() => {
  ...
});
```
通过调用tf.tidy()，我们告诉TensorFlow.js破坏Tensor我们在内部分配的中间存储器的内存capture()。有关内存管理和更多信息，请参阅 [核心概念教程tf.tidy()](./core-concepts.md#tf.tidy)

# 加载mobilenet
在我们建立模型之前，我们需要将预先训练的MobileNet加载到网页中。从这个模型中，我们将构建一个新模型，从MobileNet输出内部激活。

这是执行此操作的代码：
```js
async function loadMobilenet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
});
```
通过调用getLayer('conv_pw_13_relu')，我们将进入预训练的MobileNet模型的内部层，并构建一个新模型，其中输入是MobileNet的相同输入，但输出的层是MobileNet的中间层，名为conv_pw_13_relu。

`注意`：我们根据经验选择了这一层 - 它适用于我们的任务。一般而言，面向预训练模型末尾的层将在传递学习任务中表现更好，因为它包含输入的更高级语义特征。尝试选择另一个图层，看看它如何影响模型质量！您可以使用它model.layers来打印模型的图层。

`注意`：有关如何将Keras模型移植到TensorFlow.js的详细信息，请查看导入Keras模型教程。

## 阶段1：收集数据
游戏的第一阶段是数据收集阶段。用户将从网络摄像头保存帧并将它们与4个类中的每一个相关联：向上，向下，向左和向右。

当我们从网络摄像头收集帧时，我们将立即通过截断的MobileNet模型提供它们并保存激活张量我们不需要保存从网络摄像头捕获的原始图像，因为我们将使用的模型火车只需要这些激活作为输入。之后，当我们从网络摄像头进行预测以实际玩游戏时，我们将首先通过截断的MobileNet模型提供帧，然后通过我们的第二个模型提供截断的Mobilenet模型的输出。

我们提供了一个ControllerDataset保存这些激活的课程，以便在培训阶段使用它们。ControllerDataset有一个方法，addExample。这将通过Tensor我们截断的MobileNet 的激活调用，并label作为一个关联number。

添加新示例时，我们将保留两个Tensor代表整个数据集的内容，xs以及ys。这些将用作我们将要训练的模型的输入。

xs表示截断的MobileNet中针对所有收集的数据的所有激活，并将所有收集的数据ys的标签表示为“one hot”表示。当我们训练我们的模型时，我们将为它提供xs和的整个数据集ys。

有关one hot编码的更多详细信息，请查看[MLCC词汇表](https://developers.google.com/machine-learning/crash-course/glossary#o)。

我们来看看实现。
```js
addExample(example, label) {
  const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]), this.numClasses));

  if (this.xs == null) {
    this.xs = tf.keep(example);
    this.ys = tf.keep(y);
  } else {
    const oldX = this.xs;
    this.xs = tf.keep(oldX.concat(example, 0));

    const oldY = this.ys;
    this.ys = tf.keep(oldY.concat(y, 0));

    oldX.dispose();
    oldY.dispose();
    y.dispose();
  }
}
```
让我们解析这个功能。
```js
const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]), this.numClasses));
```
该行将对应于标签的整数转换为该标签的one hot表示。

例如，如果label = 1对应于“左”类，则one hot表示将是[0, 1, 0, 0]。我们进行这种转换，以便这代表概率分布，在类1中具有100％概率，“左”
```js
if (this.xs == null) {
  this.xs = tf.keep(example);
  this.ys = tf.keep(y);
}
```
当我们将第一个示例添加到数据集中时，我们将只保留给定的值。

我们调用tf.keep()输入Tensors，这样它们就不会被任何tf.tidy()可以包含调用的东西处理掉addExample。有关内存管理的更多信息，请参阅[CoreConcepts](./core-concepts.md)。
```js
} else {
  const oldX = this.xs;
  this.xs = tf.keep(oldX.concat(example, 0));

  const oldY = this.ys;
  this.ys = tf.keep(oldY.concat(y, 0));

  oldX.dispose();
  oldY.dispose();
  y.dispose();
}
```
当我们已经添加了一个例子，我们的数据，我们会通过调用串连新的例子来设定的现有的例子concat，用axis 的参数组来0。这会不断将我们的输入激活堆叠到xs我们的标签中ys。然后我们将处理（）任何旧的值xs和ys。

例如，如果我们的第一个标签（1）看起来像：
```js
[[0, 1, 0, 0]]
```
然后第二个电话后addExample有label = 2，ys会是这样的：
```js
[[0, 1, 0, 0],
 [0, 0, 1, 0]]
```
xs将具有相似的形状，但具有更高的维度，因为我们正在使用3D激活（制作xs4D，其中最外层的维度是收集的示例的数量）。

现在，回到index.js定义核心逻辑的地方，我们已经定义：
```js
ui.setExampleHandler(label => {
  tf.tidy(() => {
    const img = webcam.capture();
    controllerDataset.addExample(mobilenet.predict(img), label);
    // ...
  });
});
```
在这个块中，我们正在使用UI注册处理程序，以便在按下向上，向下，向左或向右按​​钮之一时进行处理，其中label对应于类索引：0,1,2或3。

在这个处理程序中，我们只是从网络摄像头捕获一个帧，通过`截断的MobileNet`提供它，生成内部激活，然后将其保存在我们的ControllerDataset对象中。

## 阶段2：训练模型
一旦用户从相关类的网络摄像头数据中收集了所有示例，我们就应该训练我们的模型！

首先，让我们设置模型的拓扑。我们将创建一个2层密集（完全连接）模型，relu在第一个密集层之后具有激活功能。
```js
model = tf.sequential({
  layers: [
    // Flattens the input to a vector so we can use it in a dense layer. While
    // technically a layer, this only performs a reshape (and has no training
    // parameters).
    tf.layers.flatten({inputShape: [7, 7, 256]}),
    tf.layers.dense({
      units: ui.getDenseUnits(),
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
      useBias: true
    }),
    // The number of units of the last layer should correspond
    // to the number of classes we want to predict.
    tf.layers.dense({
      units: NUM_CLASSES,
      kernelInitializer: 'varianceScaling',
      useBias: false,
      activation: 'softmax'
    })
  ]
});
```
您会注意到模型的第一层实际上是一个flatten图层。我们需要将输入展平为向量，以便我们可以在密集层中使用它们。inputShape展平图层的 参数对应于我们截断的MobileNet的激活形状。

我们要添加的下一层是一个密集层。我们将使用用户从UI中选择的单元对其进行初始化，使用relu 激活函数，使用varianceScaling内核初始化程序，我们将添加偏差。

我们要添加的最后一层是另一个密集层。我们将使用与我们想要预测的类数相对应的单位数来初始化它。我们将使用softmax激活函数，这意味着我们将最后一层的输出解释为可能类的概率分布。

查看[API参考](https://js.tensorflow.org/api/latest/index.html)以获取有关层构造函数参数的详细信息，或查看[用卷积神经网络识别手写数字](./mnist.md)教程。
```js
const optimizer = tf.train.adam(ui.getLearningRate());
model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
```
这是我们构建优化器，定义损失函数，编译模型以准备进行训练的地方。

我们在这里使用的是优化器，它可以很好地完成这项任务。我们的损失函数categoricalCrossentropy将测量我们4个类的预测概率分布与真实标签（one hot编码标签）之间的误差。
```js
const batchSize =
    Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
```
由于我们的数据集是动态的（用户定义要收集的数据集的大小），因此我们相应地调整批量大小。用户可能不会收集数千个示例，因此我们的批量大小可能不会太大。

现在让我们训练模型吧！
```js
model.fit(controllerDataset.xs, controllerDataset.ys, {
  batchSize,
  epochs: ui.getEpochs(),
  callbacks: {
    onBatchEnd: async (batch, logs) => {
      // Log the cost for every batch that is fed.
      ui.trainStatus('Cost: ' + logs.loss.toFixed(5));
      await tf.nextFrame();
    }
  }
});
```
model.fit可以将整个数据集作为xs和ys我们从控制器数据集传递的数据集。

我们epochs从UI 设置，允许用户定义训练模型的时间。

我们还注册了一个onBatchEnd回调fit函数，该函数在完成培训批处理的内部培训循环后调用，允许我们在模型训练时向用户显示中间成本值。我们await tf.nextFrame() 允许UI在培训期间更新。

有关此损耗性能函数的更多详细信息，请参阅[用卷积神经网络识别手写数字](./mnist.md)教程。

# 第3阶段：玩Pacman
一旦我们的模型经过培训，并且我们的成本价值已经下降，我们就可以通过网络摄像头进行预测！

这是预测循环：
```js
while (isPredicting) {
  const predictedClass = tf.tidy(() => {
    const img = webcam.capture();
    const activation = mobilenet.predict(img);
    const predictions = model.predict(activation);
    return predictions.as1D().argMax();
  });

  const classId = (await predictedClass.data())[0];
  predictedClass.dispose();

  ui.predictClass(classId);
  await tf.nextFrame();
}
```
让我们解析代码：
```js
const img = webcam.capture();
```
正如我们之前看到的，它从网络摄像头捕获一帧作为Tensor。
```js
const activation = mobilenet.predict(img);
```
现在，通过我们截断的MobileNet模型提供网络摄像头帧，以获得内部MobileNet激活。
```js
const predictions = model.predict(act);
```
现在，通过我们训练的模型提供激活，以获得一组预测。这是输出类别上的概率分布（该预测向量中的4个值中的每一个代表该类的概率）。
```js
predictions.as1D().argMax();
```
最后，压平输出，然后调用argMax。这将返回具有最高值（概率）的索引。这对应于预测的类别。
```js
const classId = (await predictedClass.data())[0];
predictedClass.dispose();

ui.predictClass(classId);
```
现在我们有一个Tensor预测标量，下载并在UI中显示！（注意我们需要Tensor在获取其值后手动处理此处，因为我们处于无法包装的异步上下文中 tf.tidy()。）

# 总结
到此为止！您现在已经学会了如何训练神经网络以从一组用户定义的类中进行预测和图像在浏览器中的处理！

如果您将此教程拷贝以进行修改，则可能必须更改模型参数才能使其适用于您的任务。