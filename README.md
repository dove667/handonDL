# handonML
这是本人在25spring写的学习笔记兼实战项目。当时我已经对ML、DL理论有了比较深入的了解，但是一直找不到好的方法练习具体的代码实现：读官方文档效率太低，网上优质视频也不多，（打脸，后面有优质视频推荐）于是决定自己边写边学。

本项目涵盖了数据集加载、清洗、预处理，模型流水线的搭建、训练和评估，模型保存和导出、部署等一整套流程。内容包括但不限于：   
- sklearn中大部分常用的API和常用的回归、分类、聚类模型和数据降维、评估metric   
- 使用pytorch从0实现的MLP，CNNs，LSTM小型神经网络  
- 基于huggingface的transformer家族：使用预训练模型（BERT）做迁移学习，huggingface的常用的model、tokenizer、pipeline、trainer等API，提示词工程（promt engineering），高效微调（PEFT）

notebook中穿插了我写代码的时候debug过程中的血泪教训和总结，特别是在不熟悉api使用规范的时候（PhaseⅠ）。因此jupyter中的代码有些散乱，我按照天数整合到了scripts中，尽量向论文提供的优质脚本靠近。

最后是关于设备。我的机器环境是windows的wsl ubuntu，cpu是CORE i5，没有独立显卡。
cpu运行sklearn中的传统机器学习模型绰绰有余，小型深度学习模型也是可以训练几个epoch的。我的wsl最大内存大约是8G，不精确估计能加载并推理的最大模型参数量约为1B（FP32）。微调模型不要用cpu。涉及微调大模型的章节我在下面的**学习规划详情**中备注上了（gpu）。
---
重要：为了让服务器的存储不爆，将ML和DL分开，本仓库只有DL，ML见另一个仓库handonML。
---

## 📋 项目结构

```text
├── data/                  # 各项目数据集       
│   ├── MNIST/             
│   ├── cifar10/                        
│   └── 
│
├── images/                # notebook中的图片
│
├── notebooks/             # Jupyter Notebook
│
├── scripts/               # 训练脚本
│
├── README.md              # 本文件
├── TODO.md                # 详细每日待办清单
```

---


# 📚 学习规划详情（旧）

## Phase I：机器学习基础 (Week 1)

### Week 1: 传统机器学习工具链
* **Day 1-2**：数据预处理与基础分类
  * 工具：`SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`
  * 模型：`LogisticRegression`
  * 评估：`classification_report`
  * 实战：UCI Adult 数据集收入预测 (目标准确率 >70%)

* **Day 3-4**：回归模型与模型优化
  * 模型：`LinearRegression`, ``XGBoost``
  * 优化：`cross_val_score`, `GridSearchCV`
  * 实战：California Housing 房价预测 (目标 R² >0.7)

* **Day 5-6**：聚类与降维
  * 降维：`PCA`
  * 模型：`KMeans` `GMM` `SpectralClustering`
  * 实战：MNIST 手写数字聚类与可视化（KMeans 聚类，t-SNE/PCA 可视化，评估聚类效果）
  
* **Day 7**： `SVC` `SVR` 比较分析支持向量机的特点

## Phase II：深度学习实战 (Week 2-3)

### Week 2: PyTorch 基础与 经典网络架构 (Day 8-14)
> [pytorch4h速通 youtube](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=1)   
* **Day 8**：`tensor` & `autograd` & `cuda` &  `Dataset` & `DataLoader`
* **Day 9** `nn.Module` / `optim ` / `tensorboard`
* **Day 10**：MLP on MNIST 架构搭建 & 训练 & 精度评估（准确率>90%）
* **Day 11**：简单 CNN on CIFAR10 架构搭建与训练（目标准确率 ≥70%）
* **Day 12**：LSTM RNN on IMDB 情感分类 (目标准确率 ≥80%)
* **Day 13**：模型可视化（tensorboard）

### Week 3: 先进网络架构 & transformer (Day 15-21)
> 本周聚焦 Huggingface Transformers 框架，深入理解 BERT、Vision Transformer (ViT)、Prompt Engineering、微调（Fine-tuning）、轻量化方法（如LoRA原理）等。

* **Day 15**：Huggingface Transformers 基础 
`Pipeline`,`AutoTokenizer`,`AutoModel`    
* **Day 16（gpu）**：Huggingface Datasets & Trainer API 进阶
`load_dataset`, `DataCollatorWithPadding`, `TrainingArguments`, `Trainer`,`evaluate`,`tqdm`
* **Day 17（gpu）**： 在SQuAD 问答任务微调bert
* **Day 18**：Prompt Engineering 入门（Zero-shot/Prompt-based 分类、文本生成）
* **Day 19**：Vision Transformer (ViT) on CIFAR-10（微调与评估，目标准确率≥85%）
* **Day 20**：预训练大模型，轻量化微调方法原理（LoRA/Adapter 理论，CPU 上小模型实验），量化
* **Day 21**：总结复盘 & 代码整理，撰写 transformer 应用经验文档

## Phase III：综合项目与前沿探索 (Week 4)

> 本阶段聚焦真实场景下的综合应用与前沿探索，涵盖文本/图像生成、微调、轻量化部署、RLHF、AutoML 等。

* **Day 22-23**：文本生成/摘要/问答系统（基于 Huggingface Pipeline，尝试 T5/BART/DistilGPT2 等小模型）
* **Day 24-25**：图像分类/检索/生成（ViT 微调、Diffusers 文生图小模型实验）
* **Day 26**：LoRA/Adapter 微调小模型实战（如 DistilBERT/DistilGPT2）
* **Day 27**：AutoML/轻量化部署（Optuna 超参搜索、ONNX 导出、CPU 推理优化）
* **Day 28**：RLHF 理论与小规模实验（如 reward model 微调、数据构造流程）
* **Day 29**：综合项目：端到端文本/图像智能应用（如“智能摘要+检索”或“文本生成+情感分析”一体化系统）
* **Day 30**：整理代码至 GitHub，撰写项目文档 & 后续规划（如强化学习、MLOps、AIGC 等）

## 📈 项目里程碑

| 周数  | 主要目标 | 验收标准 |
|-------|---------|----------|
| Week 1 | 机器学习基础 | Adult 收入预测acc >70%<br>California Housing 房价预测 R² >0.7 |
| Week 2 | PyTorch 基础 | MNIST 数字分类acc >90% <br> CIFAR10 对象分类 acc>70% <br> IMDB 情感分类 acc >80%
| Week 3 | Transformer & 先进网络 | BERT/ViT 微调准确率 ≥85%，Prompt/LoRA 理论与实验 |
| Week 4+ | 综合/前沿 | 文本/图像生成、LoRA 微调、AutoML、RLHF 等前沿项目实战 |

---
