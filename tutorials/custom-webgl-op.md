# 创建自定义WebGL操作
要定义自定义WebGL操作，我们所要做的就是创建一个实现的对象tf.webgl.GPGPUProgram。

该接口定义为：
```js
interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting?: boolean;
}
```
举个例子，让我们实现一个计算的操作f(x) = x * x + x。

这个的GLSL代码是：
```js
void main() {
    float x = getXAtOutCoords();
    float value = x * x + x;
    setOutput(value);
}
```
其中getXAtOutCoords和setOutput是由Tensorflow.js提供到着色器。请注意，为输出tensor中的每个值调用main函数。

完整的GPGPUProgram定义是：
```js
const squareAndAddKernel = inputShape => ({
  variableNames: ['X'],
  outputShape: inputShape.slice(),
  userCode: `
    void main() {
        float x = getXAtOutCoords();
        float value = x * x + x;
        setOutput(value);
      }
  `
})
```
要运行此操作，您可以使用tf.ENV.backend.compileAndRun(program: GPGPUProgram, inputs: tf.Tensor[]): tf.Tensor。请注意，如果后端不是webgl后端，则这将是未定义的。
```js
const x = tf.tensor([1, 2, 3, 4]);
const program = squareAndAddKernel(x.shape);

const result = tf.ENV.backend.compileAndRun(program, [x]);
```
但是，我们可能还想为此op定义渐变，以便渐变可以通过它反向传播。

为此，我们使用tf.customGrad。
```js
const squareAndAddBackpropKernel = inputShape => ({
  variableNames: ['X'],
  outputShape: inputShape.slice(),
  userCode: `
    void main() {
      float x = getXAtOutCoords();
      float value = 2.0 * x + 1.0;
      setOutput(value);
    }
  `
});


const squareAndAdd = tf.customGrad(x => {
  const backend = tf.ENV.backend;
  const program = squareAndAddKernel(x.shape);
  const backpropProgram = squareAndAddBackpropKernel(x.shape);

  const value = backend.compileAndRun(program, [x]);

  const gradFunc = dy =>
      [backend.compileAndRun(backpropProgram, [x]).mul(dy)];
  return {value, gradFunc};
});
```
然后我们可以使用它：
```js
const x = tf.tensor([1, 2, 3, 4]);

const value = squareAndAdd(x);

const grads = tf.grad(x => squareAndAdd(x));
const dx = grads(x);

// value == [2, 6, 12, 20]
// dx == [3, 5, 7, 9]
```
或者更简洁：
```js
const {value, grad} = tf.valueAndGrad(squareAndAdd)(x);
```
# 由Tensorflow.js生成的GLSL函数
Tensorflow.js生成可用于从输入tensor读取并写入输出tensor的函数，以及其他数字实用程序函数。这些由着色器编译器预先添加到您的代码中。

* void setOutput(float value)
    * 设置运行片段着色器的坐标的输出值（相当于gl_FragCoord = vec4(value, 0.0, 0.0, 0.0)）。
* indexType getOutputCoords()
    * 其中indexType是int | ivec2 | ivec3 | ivec4 | ivec5 | ivec6其中的一个。
    * 如果输出tensor返回int为rank-0或rank-1，否则返回ivecNN == rank。这是此程序将写入的输出tensor中的单元格的坐标。
* Tensorflow.js生成GLSL函数以从输入tensor进行采样。这些形式如下：
```js
float get{VarName}AtOutCoords()

float get{VarName}() // rank-0 input
float get{VarName}(int x) // rank-1 input
float get{VarName}(int x, int y) // rank-2 input
float get{VarName}(int x, int y, int z) // rank-3 input
float get{VarName}(int x, int y, int z, int w) // rank-4 input
// continue as above for rank-5 & rank-6

// For example, for rank-2 Tensor named x:
// float getX(int x, int y)
```
VarName是为中定义的变量名variableNames的数组GPGPUProgram中与captialised的第一个字母。这意味着对于名为的变量matrix，TF.js将生成getMatrix。

其中许多函数都取决于输入tensor的等级，所以在你的GPGPUProgram经常想要根据inputShapes 的行列发出不同的代码。例如，如果get{VarName}AtOutCoords()不存在，我们可能写成squareAndAddKernel：
```js
const squareAndAddKernel = inputShape => ({
  const variableNames = ['X']
  const outputShape = inputShape.slice()
  const rank = outputShape.length

  const coordSnippets = ['',
      'coords',
      'coords.x, coords.y',
      'coords.x, coords.y, coords.z',
      'coords.x, coords.y, coords.z, coords.w']

  const coordType = rank < 2 ? 'int' : `ivec${rank}`

  const userCode = `
    void main() {
      ${coordType} coords = getOutputCoords();
      float x = getX(${coordSnippets[rank]});
      setOutput(x * x + x);
    }`

  return {variableNames, outputShape, userCode}
})
```
* bool isNaN(float val)
    * true如果val是a NaN，否则为false。
* int round(float value)
    * 四舍五入value到最接近的整数。
* int imod(int x, int y)
    * 与float mod(float x, float y)int 相同，因为GLSL没提供给我们这样的方法。
* float random(float seed)
    * 根据Dave Hoskins在https://www.shadertoy.com/view/4djSRW 中的公式返回一个伪随机数。