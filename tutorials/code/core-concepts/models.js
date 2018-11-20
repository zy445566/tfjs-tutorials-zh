const tf = require('@tensorflow/tfjs');

// Load the binding:
// 这里可以解锁cpu性能，不安装的话命令行会有一行提示，但这个包需要翻墙，可能对初学者会有一些困难
// 安装了的话，直接解除注释即可使用
// require('@tensorflow/tfjs-node');
// 这里可以解锁gpu性能，和上面一致，不多说，电脑性能好的，可以二选一使用
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.

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