// trainSize totol 60000
// testSize totol 10000
// How many examples the model should "see" before making a parameter update.
// 每一次模型训练都要看多少示例
const BATCH_SIZE = 64;
// How many batches to train the model for.
// 训练次数
const TRAIN_BATCHES = 100;

// Every TEST_ITERATION_FREQUENCY batches, test accuracy over TEST_BATCH_SIZE examples.
// Ideally, we'd compute accuracy over the whole test set, but for performance
// reasons we'll use a subset.
// 每一次测试的张数
const TEST_BATCH_SIZE = 1000;
// 测试迭代循环频率
const TEST_ITERATION_FREQUENCY = 5;



module.exports = async function trains(model,data){
  for (let i = 0; i < TRAIN_BATCHES; i++) {
    // 获取一批培训的示例
    const batch = data.nextTrainBatch(BATCH_SIZE);
  
    let testBatch;
    let validationData;
    // Every few batches test the accuracy of the mode.
    // 每训练TEST_ITERATION_FREQUENCY次就测试一次，来确定训练的准确性
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
      //因为使用node版本的数据文件，所以无需还原形状，和官方略有修改
      validationData = [
        testBatch.images, testBatch.labels
      ];
    }
  
    // The entire dataset doesn't fit into memory so we call fit repeatedly
    // with batches.
    // 训练模型的地方
    const history = await model.fit(
        batch.images, //我们的输入图像数据,因为使用node版本的数据文件，所以无需还原形状，和官方略有修改
        batch.labels, // 我们的标签
        {
          batchSize: BATCH_SIZE, //每个培训批次中包含多少图像
          validationData, //测试的结果验证集
          epochs: 1 //批量执行的训练次数
        });
    // 损耗计算值
    const loss = history.history.loss[0];
    // 精确度精确度值
    const accuracy = history.history.acc[0];
    console.log(`No ${i} Train.loss value,accuracy value:`)
    // 按照训练理论来说损耗率将不断降低，准确度将不断提升
    console.log(`loss rate:${loss}`)
    console.log(`accuracy rate:${accuracy*100}%`);
  }
}