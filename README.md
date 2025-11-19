# handonDL — 深度学习学习笔记与实践

这是一个以深度学习为主的个人学习与实践仓库。原有的机器学习（ML）笔记已从主路径中剥离，目前仓库集中记录我在 PyTorch 与 Hugging Face 生态下的实践、实验与思考：包含 Jupyter Notebook 的探索记录与整理成脚本的训练/评估流程。

写这个仓库的初衷很简单：通过“做中学”把概念、API 与工程实践结合起来。代码中有实验记录、超参尝试、以及遇到的问题与解决办法，适合想通过实战快速理解深度学习工具链的读者参考与复现。


## 主要内容（高层说明）

- PyTorch 小型网络实现与训练脚本（MLP / CNN / RNN 等经典模型的练习版）
- 基于 Hugging Face 的 Transformer 实验（微调、数据集处理、tokenizer 用法、Trainer 用例）
- Jupyter Notebook：探索性实验、可视化与调试记录（实验思路与中间结果）
- 脚本（scripts/）: 将 notebook 中成熟或重复可运行的代码整理为脚本以便复现

注意：仓库中有些高成本（大模型微调、训练）相关内容被标注为“需要 GPU”，在没有 GPU 的环境下可使用小规模实例或读取预训练结果进行推理与分析。

## 📁 当前项目结构（简要）

```
handonDL/
├── images/         # Notebook 中的示意图与结果图
├── notebook/       # Jupyter notebooks（按天/主题组织）
├── scripts/        # 可直接运行的训练/评估脚本（按天/任务）
├── LICENSE
├── README.md
```

在本仓库中，你会看到以 dayX 命名的 notebook 与对应的脚本（例如 `notebook/day10.ipynb` 与 `scripts/day10.py`）。Notebook 用于交互式探索，scripts 用于可重复运行的流水线。

## 如何开始

1. 克隆仓库并进入目录。
2. 建议使用虚拟环境（venv/conda），安装常见依赖：PyTorch、transformers、datasets、tqdm 等。示例（macOS / zsh 下）:

```bash
python3 -m venv .venv
source .venv/bin/activate
# 然后使用 pip 安装依赖，例如：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tqdm
```

3. 运行 notebook：

```bash
jupyter lab        # 或 jupyter notebook
```

4. 运行脚本（以 CPU 为例）：

```bash
python3 scripts/day10.py
```

说明：若要在 GPU 上运行，请确保 CUDA 与相应的 PyTorch 版本匹配，并在运行脚本时指定 device 或使用 Trainer 的 `--device` 参数。

## 学习计划（保留的练习方向）

下面是仓库关联的学习任务与练习方向（按主题分组，时间节点与具体实现可能与仓库中的 notebook/script 对应）：

- PyTorch 基础与经典网络：tensor/autograd、Dataset/DataLoader、训练循环、模型保存/加载、可视化（TensorBoard）
- 小型任务实战：MNIST / CIFAR-10 / IMDB（作为练习目标，用以掌握训练/评估/调参流程）
- Transformer 应用与微调：使用 Hugging Face 的 tokenizer/Trainer、datasets，做分类/问答/生成任务的微调（标注需要 GPU 的项）
- 轻量化与工程化：LoRA/Adapter 简介、量化/导出（ONNX）、推理优化与部署思路

（仓库中 notebook 与 scripts 提供了对应的实验记录；未完成或高成本的任务会以“需要 GPU”或“未来计划”形式说明。）

## 贡献与复现建议

- 我欢迎 Issue 或 PR（清晰描述复现步骤与问题）。
- 在复现实验时，请先在小数据/小模型上验证流程，然后再放到更大规模上跑。大模型训练请在具备 GPU 的环境中进行。

## 致谢与许可

本仓库用于个人学习与分享，代码尽量保持简洁、易读，部分实现参考自官方文档与社区优秀示例，均在代码注释中注明来源。仓库采用 LICENSE 中的许可条款。



