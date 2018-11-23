# 图像训练：用卷积神经网络识别手写数字
在本教程中，我们将构建一个TensorFlow.js模型，用卷积神经网络对手写数字进行分类。首先，我们将通过“查看”成千上万的手写数字图像及其标签来训练分类器。然后我们将使用模型从未见过的测试数据来评估分类器的准确性。

# 先决条件
本教程假设您熟悉TensorFlow.js（Tensors，Variables和Ops）的基本构建，以及优化器和损失计算的概念。有关这些主题的更多背景知识，我们建议您在本教程之前完成以下教程：

* [TensorFlow.js中的核心概念](./core-concepts.md)
* [训练第一步：拟合曲线到合成数据](./fit-curve.md)

# 运行代码
可以在TensorFlow.js的[示例存储库tfjs-examples](https://github.com/tensorflow/tfjs-examples/)里面的[mnist](https://github.com/tensorflow/tfjs-examples/tree/master/mnist)目录中找到本教程的完整代码。

要在本地运行代码，您需要安装以下依赖项：
* Node.js8.9或更高版本
* Yarn 或者 NPM CLI

这些说明使用Yarn，但是如果您熟悉NPM CLI并且更喜欢使用它，那么它仍然可以使用。

您可以通过克隆repo并构建演示来运行示例的代码：

```sh
$ git clone https://github.com/tensorflow/tfjs-examples
$ cd tfjs-examples/mnist
$ yarn
$ yarn watch
```

上面的[tfjs-examples/mnist](https://github.com/tensorflow/tfjs-examples/tree/master/mnist)目录是完全独立的，因此您可以复制它以启动您自己的项目。

`注意`：本教程的代码与[tfjs-examples/mnist-core](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-core)示例之间的区别在于，我们使用TensorFlow.js的更高级API（Model，Layers）来构建模型，而[mnist-core](https://github.com/tensorflow/tfjs-examples/tree/master/mnist-core)使用更低级别的线性代数操作建立一个神经网络。

# 用于训练的数据
我们将在本教程中使用[MNIST手写数据集](http://yann.lecun.com/exdb/mnist/)。我们将学习分类的手写MNIST数字如下所示：

![mnist_4.png](./pics/mnist_4.png)
![mnist_3.png](./pics/mnist_3.png)
![mnist_8.png](./pics/mnist_8.png)

为了预处理我们的数据，我们编写了[data.js](https://github.com/tensorflow/tfjs-examples/blob/master/mnist-core/data.js)，其中包含MnistData从我们提供的MNIST数据集的托管版本中提取随机批次的MNIST图像的类。

`MnistData`将整个数据集拆分为训练数据和测试数据。当我们训练模型时，分类器将仅看到训练集。当我们评估模型时，我们将仅使用模型尚未看到的测试集中的数据来查看模型的预测对于全新数据的推广程度。

`MnistData` 有两种公共方法：
* `nextTrainBatch(batchSize)`：从训练集中返回随机批次的图像及其标签
* `nextTestBatch(batchSize)`：从测试集中返回一批图像及其标签

`注意`：在训练MNIST分类器时，随机重排数据非常重要，因此模型的预测不受我们提供图像的顺序的影响。例如，如果我们首先将模型全部输入1个数字，在此阶段的训练期间，模型可能会学会简单地预测1（因为这会使损失计算最小化）。如果我们然后仅将模型馈送2秒，它可能只是切换到仅预测2并且从不预测1（再次，这将最小化新图像集的损失）。该模型永远不会学习对有代表性的数字样本进行准确预测。

# 构建模型
在本节中，我们将构建一个卷积图像分类器模型。为此，我们将使用一个Sequential模型（最简单的模型），其中张量从一个层连续传递到下一个层。

首先，让我们Sequential用tf.sequential以下方法实例化我们的模型：
```js
const model = tf.sequential();
```
现在我们已经创建了一个模型，让我们为它添加图层。

# 添加第一层

我们要添加的第一层是二维卷积层。卷积层在图像上滑动滤镜窗口以学习空间不变的变换（即，图像的不同部分中的图案或对象将以相同的方式处理）。有关卷积的更多信息，请参阅[此文章](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)。

我们可以使用创建二维卷积层tf.layers.conv2d，它接受一个定义图层结构的配置对象：
```js
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling'
}));
```
让我们分解配置对象中的每个参数：

* inputShape。将流入模型第一层的数据形状。在这种情况下，我们的MNIST示例是28x28像素的黑白图像。图像数据的规范格式是[row, column, depth]，所以这里我们要为[28, 28, 1]每个维度中的像素数配置-28行和列的形状，深度为1，因为我们的图像只有1个颜色通道：

* kernelSize。要应用于输入数据的滑动卷积滤波器窗口的大小。在这里，我们设置kernelSize的5，它指定一个正方形，5x5的卷积窗口。

* filters。kernelSize要应用于输入数据的大小的过滤器窗口数。在这里，我们将对数据应用8个过滤器。

* strides。滑动窗口的“步长” - 即每次在图像上移动时滤波器将移动多少像素。在这里，我们指定步幅为1，这意味着滤镜将以1像素的步长滑过图像。

* activation。卷积完成后应用于数据的激活功能。在这种情况下，我们正在应用整流线性单元（ReLU）功能，这是ML模型中非常常见的激活功能。

* kernelInitializer。用于随机初始化模型权重的方法，这对训练动力学非常重要。我们不会在这里详细介绍初始化，但是VarianceScaling（这里使用）通常是一个很好的初始化器选择。

# 添加第二层
让我们在模型中添加第二层：最大池层，我们将使用它创建tf.layers.maxPooling2d。该层将通过计算每个滑动窗口的最大值来从卷积中对结果（也称为激活）进行下采样：
```js
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
```
让我们解析这些论点：
* poolSize。要应用于输入数据的滑动池窗口的大小。在这里，我们设置poolSize的[2,2]，这意味着汇集层将应用2×2窗口的输入数据。
* strides。滑动池窗口的“步长” - 即每次窗口在输入数据上移动时窗口将移动多少像素。在这里，我们指定步幅[2, 2]，这意味着滤镜将在水平和垂直方向上以2像素的步长滑过图像。

`注`：由于这两个poolSize和strides是2×2的集中窗口将完全不重叠。这意味着池化层将激活前一层的激活大小减半。

# 添加剩余的图层
重复层结构是神经网络中的常见模式。让我们添加第二个卷积层，然后添加另一个池模型到我们的模型。请注意，在我们的第二个卷积层中，我们将过滤器的数量从8增加到16.还要注意我们没有指定inputShape，因为它可以从前一层的输出形状推断：
```js
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'VarianceScaling'
}));

model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2]
}));
```
接下来，让我们添加一个`flatten`图层，将前一层的输出展平为矢量：
```js
model.add(tf.layers.flatten());
```
最后，让我们添加一个dense层（也称为完全连接层），它将执行最终分类。在密集层之前展平卷积+池层对的输出是神经网络中的另一种常见模式：
```js
model.add(tf.layers.dense({
  units: 10,
  kernelInitializer: 'VarianceScaling',
  activation: 'softmax'
}));
```
让我们分解传递给dense图层的参数。

* units。输出激活的大小。由于这是最后一层，我们正在进行10级分类任务（数字0-9），我们在这里使用10个单位。（有时单位被称为神经元的数量，但我们将避免使用该术语。）
* kernelInitializer。我们将对VarianceScaling用于卷积层的密集层使用相同的初始化策略。
* activation。分类任务的最后一层的激活功能通常是softmax。Softmax将我们的10维输出向量归一化为概率分布，因此我们有10个类别中每个类别的概率。

# 培训模型
为了实际驱动模型的训练，我们需要构造一个优化器并定义一个损失计算函数。我们还将定义评估指标，以衡量我们的模型对数据的执行情况。

`注意`：要深入了解TensorFlow.js中的优化器和损失计算函数，请参阅“[训练第一步](./fit-curve.md)”教程。

# 定义优化程序
对于我们的卷积神经网络模型，我们将使用学习率为0.15 的随机梯度下降（SGD）优化器：
```js
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
```

# 定义损失计算函数
对于我们的损失计算函数，我们将使用cross-entropy（categoricalCrossentropy,交叉熵），它通常用于优化分类任务。categoricalCrossentropy测量由模型的最后一层生成的概率分布与我们的标签给出的概率分布之间的误差，该分布将是在正确的类标签中具有1（100％）的分布。例如，给定数字7的示例的以下标签和预测值：

| 分类 | 0 | 1 |	2 |	3 |	4 |	5 |	6 |	7 |	8 |	9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 标签 | 0 | 0 | 0 |	0 |	0 |	0 |	0 |	1 |	0 |	0 |
| 预测 | .1 |	.01 |	.01 |	.01 |	.20 |	.01 |	.01 |	.60 | .03 |	.02 |

categoricalCrossentropy给出了一个较低的损失值，如果预测是高概率位是7，和一个更高的损失值，如果预测是低概率7。在训练期间，模型将更新其内部参数以最小化categoricalCrossentropy整个数据集。

# 定义评估指标
对于我们的评估指标，我们将使用准确度，该准确度衡量所有预测中正确预测的百分比。

# 编译模型
为了编译模型，我们使用我们的优化器，损失函数和评估指标列表（仅此处'accuracy'）传递一个配置对象：
```js
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});
```
# 配置批量大小
在开始培训之前，我们需要定义一些与批量大小相关的参数：
```js
// How many examples the model should "see" before making a parameter update.
// 每一次模型训练都要看多少示例
const BATCH_SIZE = 64;
// How many batches to train the model for.
const TRAIN_BATCHES = 100;

// Every TEST_ITERATION_FREQUENCY batches, test accuracy over TEST_BATCH_SIZE examples.
// Ideally, we'd compute accuracy over the whole test set, but for performance
// reasons we'll use a subset.
const TEST_BATCH_SIZE = 1000;
// 测试迭代循环频率
const TEST_ITERATION_FREQUENCY = 5;
```
<strong>有关批处理和批处理大小的更多信息</strong>

为了充分利用GPU并行计算的能力，我们希望将多个输入一起批处理，并使用单个前馈调用通过网络提供它们。

我们对计算进行批处理的另一个原因是，在优化期间，我们仅在对几个示例的梯度进行平均后才更新内部参数（采取步骤）。这有助于我们避免因错误的方向而向前迈出一步（例如，错误标记的数字）。

在对输入数据进行批处理时，我们引入了秩D + 1的张量，其中D是单个输入的维数。

如前所述，我们的MNIST数据集中单个图像的维度是[28, 28, 1]。当我们设置BATCH_SIZE64时，我们一次批量处理64个图像，这意味着我们数据的实际形状是[64, 28, 28, 1]（批处理总是最外层的维度）。

`注意`：回想一下，inputShape我们的第一个配置中conv2d没有指定批量大小（64）。配置编写为批量大小不可知，因此它们能够接受任意大小的批量。

# 编码训练循环
以下是训练循环的代码：
```js
for (let i = 0; i < TRAIN_BATCHES; i++) {
  const batch = data.nextTrainBatch(BATCH_SIZE);
 
  let testBatch;
  let validationData;
  // Every few batches test the accuracy of the mode.
  if (i % TEST_ITERATION_FREQUENCY === 0) {
    testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
    validationData = [
      testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
    ];
  }
 
  // The entire dataset doesn't fit into memory so we call fit repeatedly
  // with batches.
  const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
      batch.labels,
      {
        batchSize: BATCH_SIZE,
        validationData,
        epochs: 1
      });

  const loss = history.history.loss[0];
  const accuracy = history.history.acc[0];

  // ... plotting code ...
}
```
让我们解析代码吧。首先，我们获取一批培训示例。回想一下，我们批量示例利用GPU并行化并在进行参数更新之前平均来自许多示例的证据：
```js
const batch = data.nextTrainBatch(BATCH_SIZE);
```
每5个步骤（TEST_ITERATION_FREQUENCY我们构建validationData一个包含来自测试集的一批MNIST图像的两个元素的数组及其相应的标签。我们将使用此数据来评估模型的准确性：
```js
if (i % TEST_ITERATION_FREQUENCY === 0) {
  testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
  validationData = [
    testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]),
    testBatch.labels
  ];
}
```
model.fit 是训练模型和参数实际更新的地方。

`注意`：model.fit()在整个数据集上调用一次将导致将整个数据集上载到GPU，这可能会冻结应用程序。为避免将过多数据上传到GPU，我们建议model.fit()在for循环内调用，一次传递一批数据，如下所示：
```js
// The entire dataset doesn't fit into memory so we call fit repeatedly
// with batches.
  const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
      {batchSize: BATCH_SIZE, validationData: validationData, epochs: 1});
```
让我们再次解析这些争论：

* x。我们的输入图像数据。请记住，我们正在批量提供示例，因此我们必须告诉 fit函数我们的批次有多大。MnistData.nextTrainBatch返回具有形状[BATCH_SIZE, 784]的图像 - 所有图像的数据在长度为784（28 * 28）的1-D向量中。但是，我们的模型期望图像数据在形状中[BATCH_SIZE, 28, 28, 1]，因此我们reshape相应地。

* y。我们的标签; 每个图像的正确数字分类。

* batchSize。每个培训批次中包含多少图像。之前我们BATCH_SIZE在这里设置了64个。

* validationData。我们构建每个TEST_ITERATION_FREQUENCY（此处，5个）批次的验证集。这个数据是形状[TEST_BATCH_SIZE, 28, 28, 1]。之前，我们设置了TEST_BATCH_SIZE1000.我们将在此数据集上计算我们的评估指标（准确度）。

* epochs。批量执行的训练次数。由于我们正在迭代地为批次提供批次fit，我们只希望它一次性从该批次进行培训。

每次调用时fit，它都会返回一个富对象，其中包含我们存储的指标日志history。我们提取每次训练迭代的损失和准确性，因此我们可以在图表上绘制它们：
```js
const loss = history.history.loss[0];
const accuracy = history.history.acc[0];
```

# 查看结果！
如果您运行完整代码，您应该看到如下输出：

![mnist_learned.png](./pics/mnist_learned.png)

看起来模型正在预测大多数图像的正确数字。非常好！

# 本文代码
[点此打开代码目录,可能对比官方有一定修改，增加了自己的注释和理解](./code/mnist)