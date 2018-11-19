const tf = require('@tensorflow/tfjs');

// Load the binding:
// 这里可以解锁cpu性能，不安装的话命令行会有一行提示，但这个包需要翻墙，可能对初学者会有一些困难
// 安装了的话，直接解除注释即可使用
// require('@tensorflow/tfjs-node');
// 这里可以解锁gpu性能，和上面一致，不多说，电脑性能好的，可以二选一使用
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.

const initialValues = tf.zeros([5]); // 这个方法就是将一行5元素的形状填充0
const biases = tf.variable(initialValues); // initialize biases
biases.print(); // output: [0, 0, 0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
biases.assign(updatedValues); // update values of biases
biases.print(); // output: [0, 1, 0, 1, 0]