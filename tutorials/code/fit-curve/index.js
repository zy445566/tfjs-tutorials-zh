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

// 定义循环训练
function train(xs, ys, numIterations = 75) {
    const optimize = optimizer();

    for (let iter = 0; iter < numIterations; iter++) {
        optimize.minimize(() => {
        const predsYs = predict(xs);
        return loss(predsYs, ys);
        });
    }
}

// 生成数据函数
function generateData(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
      const [a, b, c, d] = [
        tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
        tf.scalar(coeff.d)
      ];
  
      const xs = tf.randomUniform([numPoints], -1, 1);
  
      // Generate polynomial data
      const three = tf.scalar(3, 'int32');
      const ys = a.mul(xs.pow(three))
        .add(b.mul(xs.square()))
        .add(c.mul(xs))
        .add(d)
        // Add random noise to the generated data
        // to make the problem a bit more interesting
        .add(tf.randomNormal([numPoints], 0, sigma));
  
      // Normalize the y values to the range 0 to 1.
      const ymin = ys.min();
      const ymax = ys.max();
      const yrange = ymax.sub(ymin);
      const ysNormalized = ys.sub(ymin).div(yrange);
  
      return {
        xs, 
        ys: ysNormalized
      };
    })
}

const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
const trainingData = generateData(100, trueCoefficients);
console.log("---True a,b,c,d:-0.800,-0.200,0.900,0.500")
console.log("---train Before a,b,c,d:")
a.print()
b.print()
c.print()
d.print()
train(trainingData.xs, trainingData.ys,1000);
console.log("---train After a,b,c,d:")
a.print()
b.print()
c.print()
d.print()
