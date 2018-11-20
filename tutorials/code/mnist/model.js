const tf = require('@tensorflow/tfjs');

// 创建连续型模型
const model = tf.sequential();

// 构建第一层卷积层（二维卷积层），用于学习空间不变的变换
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1], // 将流入模型第一层的数据形状
    kernelSize: 5, //要应用于输入数据的滑动卷积滤波器窗口的大小
    filters: 8, //kernelSize要应用于输入数据的大小的过滤器窗口数
    strides: 1, //滑动窗口的“步长” - 即每次在图像上移动时滤波器将移动多少像素
    activation: 'relu', //卷积完成后应用于数据的激活功能，流线性单元功能
    kernelInitializer: 'VarianceScaling' //用于随机初始化模型权重的方法，连续差额权重
}));

// 构建第二层卷积层（最大池层），用于学习空间不变的变换
model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2], // 输入数据的滑动池窗口的大小
    strides: [2, 2] //滑动池窗口的“步长”，每次窗口在输入数据上移动时窗口将移动多少像素
}));

//  重复卷积层，和上面两层相似，inputShape则是通过上一个卷积层推断，参数微调
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

// 添加展平层
model.add(tf.layers.flatten());


// 添加完全连接层，执行最终分类
model.add(tf.layers.dense({
    units: 10, // 输出等级分类
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax' // 激活功能模式，softmax将分布转换为概率分布
}));

// 定义优化器
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE); // sgd之前说过的随机梯度下降法

// 编译模型
model.compile({
    optimizer: optimizer, // 设置模型的优化器
    loss: 'categoricalCrossentropy', // 设置模型的损耗计算函数直接使用 交叉熵
    metrics: ['accuracy'], // 设置模型的评估指标
});

module.exports = model;





