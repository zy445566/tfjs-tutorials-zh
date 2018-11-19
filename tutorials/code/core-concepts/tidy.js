const tf = require('@tensorflow/tfjs');

// Load the binding:
// 这里可以解锁cpu性能，不安装的话命令行会有一行提示，但这个包需要翻墙，可能对初学者会有一些困难
// 安装了的话，直接解除注释即可使用
// require('@tensorflow/tfjs-node');
// 这里可以解锁gpu性能，和上面一致，不多说，电脑性能好的，可以二选一使用
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.

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