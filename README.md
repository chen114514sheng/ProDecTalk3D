# ProDecTalk3D：结合分阶段解耦和向量量化扩散的3D说话人脸可控生成<br>ProDecTalk3D: Controllable 3D Talking Face Generation with Progressive Decoupling and Vector Quantized Diffusion

## 简介

本仓库为论文 **《结合分阶段解耦和向量量化扩散的3D说话人脸可控生成》** 的代码实现。

ProDecTalk3D 在 VQ-VAE 的基础上引入 **分阶段解耦** 与 **向量量化扩散** 两个关键思想：一方面，在重建阶段基于身份向量、文本描述等条件对人脸特征进行分阶段解耦与反向融合，以提升特征表示质量；另一方面，在生成阶段以语音为主导，引入向量量化扩散模型，并通过分阶段加入不同模态条件，逐步约束顶层与底层人脸特征的生成范围，从而提升生成精度与情感表达稳定性。

- 论文状态：当前论文尚未投稿，仓库内容以代码与实验实现为准
- 仓库地址：https://github.com/chen114514sheng/ProDecTalk3D

## 方法概述

本文方法包含两个部分：

- **重建阶段**：分阶段解耦重建。基于身份向量、文本描述等条件对顶层与底层人脸特征进行逐步解耦与反向融合。
- **生成阶段**：条件生成建模。在重建阶段离散表示的基础上，结合语音、文本和身份条件生成人脸运动参数。

### 重建阶段

![stage1](images/图1.png)

### 生成阶段

![stage2](images/图2.png)

### ProDecTalk3D人脸生成模型

![stage3](images/图3.png)

## 项目结构

```text
ProDecTalk3D/
├── DataProcess/           # 数据预处理与数据集划分
├── FLAME/                 # FLAME 相关文件与模板
├── VQVAE2/                # 第一阶段：分阶段解耦 VQ-VAE
├── Diffusion/             # 第二阶段：向量量化扩散生成模型
├── Render.py              # 生成说话人脸视频
├── Quality.py             # 定量分析与可视化对比
├── Experiments/           # 条件交换实验（可选）
├── AuxClassifier/         # 辅助分类器（可选）
├── Utils.py               # 通用工具函数
├── config.yaml            # 路径与训练/预测配置
└── environment.yml        # 环境依赖参考
```

## 运行环境

建议使用 Linux + NVIDIA GPU 环境运行。

项目环境建议以 `environment.yml` 为参考，并结合本地 CUDA 与 PyTorch 版本进行配置。

## 数据集

本项目使用：

- **MEAD**：语音与视频数据  
  官网：https://wywu.github.io/projects/MEAD/MEAD.html
- **3DMEAD**：基于 MEAD 进一步处理得到的 3D 人脸运动数据  
  参考处理来源：https://github.com/radekd91/inferno/tree/release/EMOTE/inferno_apps/TalkingHead/data_processing
- **TA-MEAD**：用于描述面部情感与动作的文本数据

### 数据预处理

```bash
python DataProcess/mead0.py
python DataProcess/mead1.py
```

## 配置说明

项目中的数据路径、FLAME 路径、模型权重路径和训练参数统一通过 `config.yaml` 设置。

至少建议检查以下内容：

- `train_file_path`
- `val_file_path`
- `test_file_path`
- `flame_model`
- `static_landmark_embedding`
- `dynamic_landmark_embedding`
- `predict.vqvae_dir`
- `predict.diffusion_dir`
- `predict.save_path`
- `stage1.checkpoint_dir`
- `stage2.checkpoint_dir`

## 训练

### 第一阶段：训练分阶段解耦 VQ-VAE

```bash
python VQVAE2/Train.py
```

### 第二阶段：训练向量量化扩散生成模型

```bash
python Diffusion/Train.py
```

## 测试与生成

### 第一阶段：评估重建效果

```bash
python VQVAE2/Predict.py
```

### 第二阶段：评估生成效果并保存结果

```bash
python Diffusion/Predict.py
```

### 渲染视频与定量分析

```bash
python Render.py
python Quality.py
```

## 条件交换实验（可选）

若希望进一步验证模型的条件解耦能力与控制能力，可使用 `Experiments/` 中的脚本：

```bash
python Experiments/build_swap_pairs.py
python Experiments/run_stage1_swap.py --pair_type text_emotion --deduplicate_reverse
python Experiments/run_stage2_swap.py --pair_type text_emotion --deduplicate_reverse
python Experiments/eval_swap_metrics.py
python Experiments/render_swap_vis.py --stage all --pair_type all
python Experiments/render_swap_video.py --stage all --pair_type all
```

若需要交换实验中的定量评估，请先训练辅助分类器：

```bash
python AuxClassifier/train_emotion.py
python AuxClassifier/train_identity.py
```

## 引用

如果本项目对你的研究有帮助，可引用对应论文与本仓库代码。

```bibtex
@misc{prodectalk3d,
  title  = {ProDecTalk3D: Controllable 3D Talking Face Generation with Progressive Decoupling and Vector Quantized Diffusion},
  author = {Chen, Sheng and Sun, Qiang},
  note   = {Unpublished manuscript and code available at https://github.com/chen114514sheng/ProDecTalk3D}
}
```
