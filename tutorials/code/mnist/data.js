const tf = require('@tensorflow/tfjs');
const assert = require('assert');
const fs = require('fs');
const util = require('util');
const path = require('path');

const readFile = util.promisify(fs.readFile);

// MNIST data constants:
const TRAIN_IMAGES_FILE = path.join(__dirname,'train-images-idx3-ubyte');
const TRAIN_LABELS_FILE = path.join(__dirname,'train-labels-idx1-ubyte');
const TEST_IMAGES_FILE = path.join(__dirname,'t10k-images-idx3-ubyte');
const TEST_LABELS_FILE = path.join(__dirname,'t10k-labels-idx1-ubyte');
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;


function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(filename) {
  const buffer = await readFile(filename);

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // Normalize the pixel values into the 0-1 interval, from
      // the original 0-255 interval.
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  assert.equal(images.length, headerValues[1]);
  return images;
}

async function loadLabels(filename) {
  const buffer = await readFile(filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);
  return labels;
}

/** Helper class to handle loading training and test data. */
class MnistDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  /** Loads training and test data. */
  async loadData() {
    this.dataset = await Promise.all([
      loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE),
      loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  nextTrainBatch(size) {
    return this.getData_(true,size);
  }

  nextTestBatch(size) {
    return this.getData_(false,size);
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  // 这个函数修改使用随机图片
  getData_(isTrainingData, size = -1) {
    let imagesIndex;
    let labelsIndex;
    if (isTrainingData) {
      imagesIndex = 0;
      labelsIndex = 1;
    } else {
      imagesIndex = 2;
      labelsIndex = 3;
    }
    size = size==-1 || size > this.dataset[imagesIndex].length?this.dataset[imagesIndex].length:size;

    // Only create one big array to hold batch of images.
    // 一组图像为什么是4维？
    // 第一个变量有多少张图
    // 第二和第三分别代表高和宽[其中高宽维度存储点位]
    // 第四个变量表示深度
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      let randNum = Math.floor(Math.random()*size);
      images.set(this.dataset[imagesIndex][randNum], imageOffset);
      labels.set(this.dataset[labelsIndex][randNum], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

module.exports = new MnistDataset();