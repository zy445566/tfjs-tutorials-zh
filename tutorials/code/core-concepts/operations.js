const tf = require('@tensorflow/tfjs');

// Load the binding:
// 这里可以解锁cpu性能，不安装的话命令行会有一行提示，但这个包需要翻墙，可能对初学者会有一些困难
// 安装了的话，直接解除注释即可使用
// require('@tensorflow/tfjs-node');
// 这里可以解锁gpu性能，和上面一致，不多说，电脑性能好的，可以二选一使用
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.

const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[1, 4 ],
//          [9, 16]]

const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// Output: [[6 , 8 ],
//          [10, 12]]