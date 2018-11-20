# TensorFlow.js中的核心概念
TensorFlow.js是一个用于机器智能的开源WebGL加速JavaScript库。它为您的指尖带来了高性能的机器学习构建块，允许您在浏览器中训练神经网络或在推理模式下运行预先训练的模型。有关安装/配置TensorFlow.js的指南，请参阅“[入门](https://js.tensorflow.org/index.html#getting-started)”。

TensorFlow.js提供用于机器学习的低级构建块以及用于构建神经网络的高级Keras启发式API。我们来看看该库的一些核心组件。

# Tensors
TensorFlow.js中的中心数据单位是tensor：一组数值，形状为一个或多个维度的数组。一个Tensor实例有一个shape定义该阵列形状属性（即，有多少个值是在所述阵列的每一维）。

Tensor构造函数是tf.tensor的函数

```js
// 2x3 Tensor
const shape = [2, 3]; // 2 rows, 3 columns
const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
a.print(); // print Tensor values
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

// The shape can also be inferred: # 这个shape也能被推测
const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
b.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]
```
然而，构建低等级的tensors，我们建议您使用以下功能来提高代码的可读性：tf.scalar，tf.tensor1d，tf.tensor2d，tf.tensor3d和tf.tensor4d。

下面这个例子创建了一个和上面完全相同的tensor，但使用了tf.tensor2d:
```js
const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
c.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]
```

TensorFlow.js还提供了方便的功能，用于创建tensors，所有值设置为0（tf.zeros）或所有值设置为1（tf.ones）：
```js
// 3x5 Tensor with all values set to 0
const zeros = tf.zeros([3, 5]);
// Output: [[0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0]]
```

在TensorFlow.js中，tensors是不可变的; 一旦创建，您就无法更改其值。取代是执行生成新tensors的操作。

# Variables
Variables是被tensor的值初始化的。与Tensors不同，它们的值是可变的。您可以使用assign方法为现有变量指定新的tensor：
```js
const initialValues = tf.zeros([5]); // 这个方法就是将一行5元素的形状填充0
const biases = tf.variable(initialValues); // initialize biases
biases.print(); // output: [0, 0, 0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
biases.assign(updatedValues); // update values of biases
biases.print(); // output: [0, 1, 0, 1, 0]
```

Variables主要用于在模型训练期间存储和更新值。

# Operations (Ops)
虽然tensors允许您存储数据，但Operations (Ops)允许您操作该数据。TensorFlow.js提供了多种适用于线性代数和机器学习的运算，可以在tensors上执行。因为tensors是不可变的，所以这些操作不会改变它们的值，而是返回新的tensors。

可用的Ops包括一元操作，例如square(平方)：
```js
const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[1, 4 ],
//          [9, 16]]
```

二进制Ops像 add（加），sub（减）以及mul（乘）：
```js
const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// Output: [[6 , 8 ],
//          [10, 12]]
```

TensorFlow.js有一个可链接的API; 你可以在ops的结果上调用ops：

```js
const sq_sum = e.add(f).square();
sq_sum.print();
// Output: [[36 , 64 ],
//          [100, 144]]

// All operations are also exposed as functions in the main namespace,
// so you could also do the following:
const sq_sum = tf.square(tf.add(e, f));
```
# Models（模型） 和 Layers（图层）
`之前Tensors,Variables,Operations (Ops)不翻译成中文是因为容易造成迷惑`

从概念上讲，模型是一种函数，给定一些输入将产生一些所需的输出。

在TensorFlow.js中，有两种方法可以创建模型。您可以直接使用ops来表示模型所做的工作。例如：
```js
// Define function
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // More on tf.tidy in the next section
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// Define constants: y = 2x^2 + 4x + 8
// 这里定义的a,b,c 就是定义的y = a * x ^ 2 + b * x + c的a,b,c。
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

// Predict output for input of 2
// 这里的输入值就是x
const result = predict(2);
// 这里的输出值就是y
result.print() // Output: 24
```

您还可以使用高级API tf.model来构建图层的模型，这是深度学习中的流行抽象。以下代码构造了一个tf.sequential模型：

```js
// 个人觉得model和layers有点像大脑和神经(比如听觉神经，视觉神经)或部分神经的关系
// 比如下面就像大脑处理一个[80, 4]的形状的神经
const model = tf.sequential();
model.add(
  tf.layers.simpleRNN({
    units: 20, // 空间维数
    recurrentInitializer: 'GlorotNormal', // 线性规则设置为正态分布
    inputShape: [80, 4] // 定义阵列形状属性为80行，每行4个元素
  })
);

const optimizer = tf.train.sgd(LEARNING_RATE);//sgd 表示随机梯度下降法,有兴趣可搜索
// 这里是编译优化器和性能损失函数，不明白没关系下节会讲
model.compile({optimizer, loss: 'categoricalCrossentropy'});
// 适配数据,用于训练模型和更新参数
model.fit({x: data, y: labels});
```

TensorFlow.js中有许多不同类型的层。举几个例子包括tf.layers.simpleRNN，tf.layers.gru，和tf.layers.lstm。

# 内存管理工具: dispose 和 tf.tidy

由于TensorFlow.js使用GPU来加速数学运算，因此在使用Tensors和Variables时需要管理GPU内存。

TensorFlow.js提供了两个函数来帮助解决这个问题：dispose和tf.tidy。

## dispose
您可以调用dispose来清除Tensors和Variables并释放其GPU内存：
```js
const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
const x_squared = x.square();

x.dispose();
x_squared.dispose();
```

## tf.tidy

dispose在进行大量Tensors操作时使用可能很麻烦。TensorFlow.js提供了另一个函数，tf.tidy它在JavaScript中扮演与常规作用域类似的角色，只不过是对于作用域于Tensors的GPU的支持。

tf.tidy执行一个函数并清除所有创建的中间张量，释放它们的GPU内存。但它不会清除内部函数的返回值。
```js
// tf.tidy takes a function to tidy up after
const average = tf.tidy(() => {
  // tf.tidy will clean up all the GPU memory used by tensors inside
  // this function, other than the tensor that is returned.
  //
  // Even in a short sequence of operations like the one below, a number
  // of intermediate tensors get created. So it is a good practice to
  // put your math ops in a tidy!
  const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
  const z = tf.ones([4]);// [4]表示一个1行4元素的矩阵，ones是将矩阵所有的值填充为1
  // 即第一步 [1.0, 2.0, 3.0, 4.0]减[1, 1, 1, 1]
  // 第二步 [0, 1, 2, 3] 平方后 [0, 1, 4, 9]
  // 第三步 求平均值 即 (0+1+4+9)/4 = 3.5
  // 这也是 均方误差的计算方法
  // 当然这里主要要表达是在tf.tidy内tensor都会被回收掉，就不用像dispose这么麻烦
  return y.sub(z).square().mean();
});

average.print() // Output: 3.5
```

使用tf.tidy将有助于防止应用程序中的内存泄漏。它还可以用于更加谨慎地控制何时回收内存。

`两个重要的注释`
* 传递给的函数tf.tidy应该是同步的，也不能返回Promise。我们建议将更新UI或在远程请求的代码放在tf.tidy函数的外面。
* tf.tidy 不会清理变量。变量通常持续到机器学习模型的整个生命周期，因此TensorFlow.js即使它们是在一个中创建的，也不会清理它们tidy。但是，您可以dispose手动调用它们。

# 本文代码
[点此打开代码目录,可能对比官方有一定修改，增加了自己的注释和理解](./code/core-concepts/)
