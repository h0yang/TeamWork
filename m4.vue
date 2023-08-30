<template>
  <div class="container">
    <div id="title">支持向量机与随机森林算法实现</div>
    <h2>数据集选择</h2>
    <div class="content">
      <el-select v-model="selectedDataset" placeholder="选择数据集">
        <el-option
          v-for="dataset in datasets"
          :key="dataset.value"
          :label="dataset.label"
          :value="dataset.value"
        ></el-option>
      </el-select>
      <el-switch v-model="preprocessingEnabled" active-text="开启预处理" inactive-text="关闭预处理"></el-switch>
      <div id="rate">数据分割比例</div>
      <div class="slider-container">
        <div class="slider-tooltip">{{ sliderValue }}%</div>
        <el-slider v-model="sliderValue" :show-tooltip="true" :min="0" :max="100" :step="1" style="width: 300px;"></el-slider>
      </div>
      <h2>模型选择</h2>
      <div class="content">
        <el-radio-group v-model="selectedModel">
          <el-radio label="SVM">SVM</el-radio>
          <el-radio label="RANDOMFOREST">Random Forest</el-radio>
        </el-radio-group>
      </div>

      <h2>划分方式</h2>
      <div class="content1">
        <el-radio-group v-model="selectedmethod">
          <el-radio label="RANDOM">random</el-radio>
          <el-radio label="KFOLD">k-fold</el-radio>
          <el-radio label="STRATIFIEDKFOLD">Stratified</el-radio>
        </el-radio-group>
      </div>

      <el-button @click="sendData" type="primary">发送数据到后端</el-button>
    </div>
    <router-link to="/"></router-link>
    <div id="back"><button @click="navigatorToapp">Back</button></div>
  </div>
</template>
<style scoped>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  height: 100vh;
  margin-bottom: 20px;
}

h2 {
  margin-bottom: 20px;
  margin-top: 20px;
}

.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;
}

.el-select {
  margin-bottom: 10px;
}

.el-button {
  margin-top: 10px;
  width: 160px;
}

.slider-container {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.slider-tooltip {
  margin-right: 10px;
  min-width: 40px;
  text-align: right;
}

#rate {
  font-size: 22px;
  font-weight: bold;
  margin-top: 22px;
  margin-bottom: 10px;
}
#title{
  color: black;
  font-size: 30px;
  font-weight: 800;
}

#back{
    position: fixed;
    top: 90%;
    left: 90%;
  }
</style>
<script>
import axios from 'axios';

export default {
  data() {
    return {
      selectedDataset: '', // 选择的数据集
      datasets: [ // 数据集列表
        { label: 'iris', value: 'iris' },
        { label: 'diabetes', value: 'diabetes' },
        { label: 'boston', value: 'boston' },
        // 添加更多数据集选项...
      ],
      preprocessingEnabled: false, // 是否进行数据预处理的开关状态
      sliderValue: 50, // 数据分割比例的初始值
      selectedModel: '', // 选择的模型
      selectedmethod:"",
      splits0:5
    };
  },

  methods: {
    sendData() {
      const data = {
        dataset: this.selectedDataset, // 选择的数据集值
        preprocessingEnabled: this.preprocessingEnabled, // 是否进行数据预处理
        split_ratio: this.sliderValue, // 数据分割比例的值
        model: this.selectedModel, // 选择的模型
        split_method:this.selectedmethod,
        n_splits:this.splits0
      };

      axios.post('http://127.0.0.1:5000/get_data', data, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
        .then(response => {
          // 请求成功处理
          console.log(response.data);
        })
        .catch(error => {
          // 请求错误处理
          console.error(error);
        });
    },
    navigatorToapp() {
      this.$router.push('/');
    },
  }
}
</script>