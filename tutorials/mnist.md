# 图像训练：用卷积神经网络识别手写数字
在本教程中，我们将构建一个TensorFlow.js模型，用卷积神经网络对手写数字进行分类。首先，我们将通过“查看”成千上万的手写数字图像及其标签来训练分类器。然后我们将使用模型从未见过的测试数据来评估分类器的准确性。

# 先决条件
本教程假设您熟悉TensorFlow.js（Tensors，Variables和Ops）的基本构建，以及优化器和损耗计算的概念。有关这些主题的更多背景知识，我们建议您在本教程之前完成以下教程：

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

`注意`：在训练MNIST分类器时，随机重排数据非常重要，因此模型的预测不受我们提供图像的顺序的影响。例如，如果我们首先将模型全部输入1个数字，在此阶段的训练期间，模型可能会学会简单地预测1（因为这会使损耗计算最小化）。如果我们然后仅将模型馈送2秒，它可能只是切换到仅预测2并且从不预测1（再次，这将最小化新图像集的损失）。该模型永远不会学习对有代表性的数字样本进行准确预测。

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