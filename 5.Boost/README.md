## Boost
## (提升算法)

### 项目背景
> boost算法是基于PAC学习理论（probably approximately correct）而建立的一套集成学习算法(ensemble learning)。其根本思想在于通过多个简单的弱分类器，构建出准确率很高的强分类器，PAC学习理论证实了这一方法的可行性。提升方法思路：对于一个复杂的问题，将多个专家的判断进行适当的综合所得出的判断，要比任何一个专家单独判断好。每一步产生一个弱预测模型(如决策树)，并加权累加到总模型中，可以用于回归和分类问题；如果每一步的弱预测模型生成都是依据损失函数的梯度方向，则称之为梯度提升(Gradient boosting)。

### 项目简介
|名称|简介|
|:-------------|:-------------:|
|5.1 xgBoost_Intro|基于XGBoost算法预测蘑菇毒性|
|5.2 xgBoost_Predict|基于XGBoost算法预测莺尾花种类|
|5.3 xgBoost_Wine|对比Logistic与XGBoost算法对于酒类数据预测|
|5.4 xgBoost_ReadData|基于XGBoost算法预测落叶松类别|
|5.5 Titanic|基于XGBoost算法预测泰坦尼克号存活率|
|5.6 Bagging_intro|Bagging操作引入以及数据模拟对比|

### 效果图
#### ·几种算法回归效果比对
<img width="550" height="400" src="./figures/bagging.png"/>

