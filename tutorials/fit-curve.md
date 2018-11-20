# 训练第一步：拟合曲线到合成数据
在本教程中，我们将使用TensorFlow.js将曲线拟合到合成数据集。给定一些使用添加了一些噪声的多项式函数生成的数据，我们将训练模型以发现用于生成数据的系数。

# 先决条件
本教程假设您熟悉TensorFlow.js的基本构建,包括在TensorFlow.js中的核心概念的Tensors，Variables和Ops。我们建议在完成本教程之前完成的[核心概念](./core-concepts.md)。

# 运行代码
本教程重点介绍用于构建模型和学习其系数的TensorFlow.js代码。可以在此处找到本教程的完整代码（包括数据生成和图表绘图代码）。

要在本地运行代码，您需要安装以下依赖项：
* Node.js8.9或更高版本
* Yarn 或者 NPM CLI

这些说明使用Yarn，但是如果您熟悉NPM CLI并且更喜欢使用它，那么它仍然可以使用。
```sh
$ git clone https://github.com/tensorflow/tfjs-examples
$ cd tfjs-examples/polynomial-regression-core
$ yarn # 对应 npm install
$ yarn watch # 对应 npm run watch
```
上面的tfjs-examples/polynomial-regression-core目录是完全独立的，因此您可以复制它以启动您自己的项目。

# 输入数据
我们的合成数据集由x坐标和y坐标组成，在笛卡尔平面上绘制时如下所示：

![fit_curve_data.png](./pics/fit_curve_data.png)

使用格式y = a*x^3 + b*x^2 + c*x + d 的三次函数生成该数据。
我们的任务是学习该函数的系数：最适合数据的a，b，c和d的值。让我们看一下如何使用TensorFlow.js操作学习这些值。

# 第1步：设置变量
首先，让我们创建一些变量来保持我们在模型训练的每个步骤中对这些值的当前最佳估计。首先，我们将为每个变量分配一个随机数：
```js
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
```

# 第2步：构建模型

我们可以通过链接一系列数学运算来表示我们在TensorFlow.js中的多项式函数y = a*x^3 + b*x^2 + c*x + d：加法（add），乘法（mul）和求幂（pow和square）。

以下代码构造一个作为输入并返回的predict函数：输入x，返回y

```js
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}
```

让我们继续使用我们在步骤1中设置的a，b，c和d的随机值绘制我们的多项式函数。我们的图可能看起来像这样：

![fit_curve_random.png](./pics/fit_curve_random.png)

因为我们从随机值开始，所以我们的函数很可能不适合数据集。该模型尚未学习更好的系数值。

# 第3步：训练模型
我们的最后一步是训练模型以学习系数的良好值。为了训练我们的模型，我们需要定义三件事：
* 一个损耗计算函数，是衡量一个给定的多项式吻合程度的数据。损耗值越低，多项式拟合数据越好。
* 一个优化器，它实现的算法用于基于所述损失函数的输出修改我们的系数值。优化器的目标是减少`损耗计算函数`的输出值。
* 一个训练循环，这将反复运行优化，以尽量减少损失。

# 定义损耗计算函数
对于本教程，我们将使用均方误差（MSE,均方误差计算方法在上节在[core-concepts.md#tf.tidy](./core-concepts.md#tftidy)说过）作为我们的损失函数。通过对我们数据集中每个x值的实际y值和预测y值之间的差值求平方，然后取所有结果项的均值来计算MSE 。

我们可以在TensorFlow.js中定义一个MSE`损耗计算函数`，如下所示：
```js
function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}
```

# 定义优化器
对于我们的优化器，我们将使用 随机梯度下降法（SGD，即从数据集中随机均匀选择的单个样本来计算每步的梯度估算值）。SGD通过获取数据集中随机点的梯度并使用其值来通知是否增加或减少模型系数的值来工作。

TensorFlow.js为执行SGD提供了便利功能，因此您不必担心自己执行所有这些数学运算(换句话说，概念你至少要知道)。tf.train.sgd将所需的学习速率作为输入，并返回一个SGDOptimizer对象，可以调用该对象以优化损失函数的值。

该学习速率控制有多大改善其预测当模特的调整会。低学习率将使学习过程运行得更慢（学习好系数需要更多的训练迭代），而高学习率将加速学习，但可能导致模型围绕正确值振荡，总是过度校正。

以下代码构造了一个学习率为0.5的SGD优化器：
```js
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
```

# 定义训练循环
现在我们已经定义了损失函数和优化器，我们可以构建一个训练循环，迭代地执行SGD来优化我们的模型系数以最小化损失（MSE）。这是我们的循环：
```js
function train(xs, ys, numIterations = 75) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}
```

让我们一步一步地仔细研究代码。首先，我们定义训练函数，将数据集的x和y值以及指定的迭代次数作为输入：

```js
function train(xs, ys, numIterations) {
...
}
```
接下来，我们定义学习速率和SGD优化器，如上一节所述：
```js
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
```
最后，我们建立了一个for运行numIterations训练迭代的循环。在每次迭代中，我们调用minimize优化器，这是魔术发生的地方：
```js
for (let iter = 0; iter < numIterations; iter++) {
  optimizer.minimize(() => {
    const predsYs = predict(xs);
    return loss(predsYs, ys);
  });
}
```
minimize 采取做两件事的功能：
1. 它使用我们之前在步骤2中定义的预测模型函数预测所有x值的y值（predYs)
2. 它使用我们之前在`定义损耗计算函数`中使用损耗计算函数返回那些预测的均方误差。

minimize然后通过Variable该功能自动调整（系数a，b，c，和d），以尽量减少返回值（我们的损耗）。

运行我们的训练循环后a，b，c，和d将包含SGD后75次迭代由模型学到的系数值。

# 看看结果！
一旦程序完成运行，我们可以采取的最终值我们的变量a，b，c，和d，并利用它们来绘制曲线：

![fit_curve_learned.png](./pics/fit_curve_learned.png)

结果比我们最初使用系数的随机值绘制的曲线好得多。

# 本文代码
[点此打开代码目录,可能对比官方有一定修改，增加了自己的注释和理解](./code/fit-curve)