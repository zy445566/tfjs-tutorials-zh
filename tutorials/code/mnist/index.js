const data = require('./data');
const model = require('./model');
const trains = require('./trains');
// 开始训练并打印结果
(async ()=>{
    console.log('start load data...');
    await data.loadData();
    console.log('load data over...');
    console.log('start trains data...');
    await trains(model,data);
})()