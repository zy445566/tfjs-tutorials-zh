const tf = require('@tensorflow/tfjs');

// Load the binding:
// 这里可以解锁cpu性能，不安装的话命令行会有一行提示，但这个包需要翻墙，可能对初学者会有一些困难
// 安装了的话，直接解除注释即可使用
// require('@tensorflow/tfjs-node');
// 这里可以解锁gpu性能，和上面一致，不多说，电脑性能好的，可以二选一使用
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.

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

const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
c.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

// 3x5 Tensor with all values set to 0
const zeros = tf.zeros([3, 5]);
// Output: [[0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0]]