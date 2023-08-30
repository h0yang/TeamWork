<template>
    <div id="demo">
      <router-link to="/"></router-link>
      <router-link to="/m1"></router-link>
      <router-link to="/m2"></router-link>
      <router-link to="/m3"></router-link>
      <router-link to="/m4"></router-link>
      <div id="title">十大算法的实现</div>
      <div id="member">小组成员：王杰 胡杨 文馨 郑祎</div>
      <div id="alg">
      <el-row>
      <el-button type="success" @click="navigatorToapp1">1.线性回归算法 8.K均值聚类算法10.梯度增强算法</el-button>
      <el-button type="success" @click="navigatorToapp2">2.逻辑回归算法 4.朴素贝叶斯算法 9.降维算法</el-button>
      <el-button type="success" @click="navigatorToapp3">3.决策树算法5.K最近邻算法</el-button>
      <el-button type="success" @click="navigatorToapp4">6.支持向量机算法 7.随机森林算法</el-button>
      </el-row>

      <div id="clock">
        <div id="time">{{time}}</div>
        <div id="date">{{date}}</div>
      </div>
    
  </div>
    </div>
 
  </template>
   <style scoped>
  #demo{
    
    position: fixed;
    height: 100%;
    width:100%;
    top:0;
    left:0;
    background-image: linear-gradient(
      90deg,cyan,purple
    );
    background-size:400%;
    animation: myanimation 10s infinite;
}
@keyframes myanimation{
  0%{
    background-position: 0% 50%;
  }
  50%{
    background-position: 100% 50%;
  }
  100%{
    background-position: 0% 50%;
  }
}
#title{
  color: aliceblue;
  font-size: 30px;
  font-weight: 800;
}
#member{
  color: black;
  font-size: 30px;
  font-weight: 800;
}
#alg{
  position: fixed;
  top: 40%;
  left: 20%;
}
#clock{
  color: aliceblue;
  font-size: 30px;
  font-weight: 800;
}
  </style>

  <script>
export default{
  data(){
return{
  time:"",
  date:"",
};
  },
mounted(){
 this.$nextTick(()=>{
setInterval(this.update_clock,1000);
 })
},
methods:{
  Linear:function(e){
    this.$axios({
          url: "http://127.0.0.1:5000/Linear",
          method: "post",
          data: {
           
          },
        })
          .then((res) => {
            console.log(res.data);
            if(res.data=="Linear"){
              this.$router.push("/Linear")
            }else{
              alert("跳转失败")
            }
          })
          .catch((err) => {
            console.log(err);
          });
  },

update_clock:function(e){
var d=new Date();
 var year=d.getFullYear();
 if(year<10){
 year="0"+year
}
var mon=d.getMonth()+1;
 if(mon<10){
 mon="0"+mon
 }
 var day=d.getDay()+16;
 if(day<10){
 day="0"+day
 }
var hour=d.getHours();
if(hour<10){
 hour="0"+hour
}
var minute=d.getMinutes();
this.time=hour+":"+minute;
this.date=year+"/"+mon+"/"+day;

},

  navigatorToapp1() {
      this.$router.push('/m1');
    },

  navigatorToapp2() {
      this.$router.push('/m2');
    },

  navigatorToapp3() {
      this.$router.push('/m3');
    },

  navigatorToapp4() {
      this.$router.push('/m4');
    },
}
}
</script>