const tf = require('@tensorflow/tfjs');

// 初始化变量a,b,c,d
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// 构建算法的模型
function predict(x) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(3))) // a * x^3
        .add(b.mul(x.square())) // + b * x ^ 2
        .add(c.mul(x)) // + c * x
        .add(d); // + d
    });
}

// 定义损耗函数，使用MSE(均方误差)来检测误差
function loss(predictions, labels) {
    // Subtract our labels (actual values) from predictions, square the results,
    // and take the mean.
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

// 定义优化器
function optimizer(predictions, labels) {
    const learningRate = 0.5;
    return tf.train.sgd(learningRate);
}

// 未完待续