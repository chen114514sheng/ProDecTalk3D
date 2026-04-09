# DecTalk3D：分层解耦引导的情感可控VQ-VAE 3D说话人脸生成方法
## 简介
项目位置：https://github.com/chen114514sheng/DecTalk3D

本项目使用语音、文本、身份向量这3个条件，在VQ-VAE结构的基础上，通过分层解耦的方法生成说话人脸视频。
## 方法概述
项目整体流程如下：

第一阶段
<img width="16538" height="11675" alt="图1" src="https://github.com/user-attachments/assets/fbbe08b7-ff75-42a8-a5d1-37eb2983673c" />
第二阶段
<img width="14813" height="8988" alt="图2" src="https://github.com/user-attachments/assets/1bd8c91b-e73f-4c1c-9ecc-9e271ae89e62" />
## 数据集
3D人脸数据来自3DMEAD，语音数据来自MEAD，文本数据来自TA-MEAD。

数据集预处理：
```
DataProcess/mead0.py
```
数据集划分：
```
DataProcess/mead1.py
```
## 训练过程
训练第一阶段模型：
```
VQVAE2/Train.py
```
训练第二阶段模型：
```
Generation/Train.py
```
## 测试与生成
评估第一阶段模型的重建效果：
```
VQVAE2/Predict.py
```
评估第二阶段模型的生成效果，保存生成结果：
```
Genetion/Predict.py
```
生成说话人脸视频：
```
Render0.py #也可用于生成其他模型的说话人脸视频
```
本模型与其他模型比较：
```
Render1.py #合成对比视频
Quality.py #生成heatmap
```
## 注意事项
模型参数、权重保存路径、文件路径通过以下文件设置：
```
config.yaml
```
以下文件中的路径同样需要设置：
```
DataProcess/mead0.py
DataProcess/mead1.py
Render0.py
Render1.py
Quality.py
```
