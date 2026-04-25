# From-Zero-to-Build-LLM

从零开始手撕预训练一个大语言模型，包括简单的复现[MiniMind项目](https://github.com/jingyaogong/minimind)，进一步训练1B级别的模型，对标GPT2，在一些指标上超越GPT2(1.5B)

# 项目结构

```
├── dataset
├── images
├── model
│   ├── Embedding.py
│   ├── FFN.py
│   ├── GQA.py
│   ├── MoE.py
│   ├── RMSNorm.py
│   ├── RoPE.py
│   └── model.py
├── .gitignore
├── README.md
├── generate_project_tree.py
├── requirements.txt
└── update_req.sh
```

# 复现MiniMind

## 环境和服务器配置
### 本人的服务器配置（供参考）
- 8卡NVIDIA GeForce RTX 3090
    - NVIDIA-SMI 535.154.05
    - Driver Version: 535.154.05
    - CUDA Version: 12.2 
### 环境安装
```bash
# 安装支持 CUDA 11.8 的 GPU 版 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
检测是否成功安装
```Python
import torch
print(torch.cuda.is_available())  # 输出 True 就是成功
print(torch.cuda.device_count())  # 显卡数量（8）
print(torch.cuda.get_device_name(0))  # 显示显卡名称（NVIDIA GeForce RTX 3090）
```

## RMSNorm
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\dfrac{1}{N}\sum\limits_{i=1}^{N} x_i^2 + \epsilon } }  \cdot \gamma 
$$

## 模型细节和参数配置
<!-- 无Flash-Attention，使用MoE，GQA， -->



# Screen训练+日志log
1. 新建一个后台窗口（训练专用）
bash运行`screen -S train3090`，
train3090 自定义名字，方便区分
2. 进入窗口后，正常操作 bash运行

cd /你的github项目路径

conda activate 你的环境名

python train.py

## 结合 训练日志 + screen 最佳组合
方案 A：screen 实时看打印 + 额外保存日志文件
bash
运行
# 在screen窗口内执行
python train.py > log/train.log 2>&1
屏幕实时输出
同时全部写入日志文件
随时看历史日志：cat log/train.log / tail -f log/train.log
方案 B：项目自带 logging/tensorboard
不用改命令，正常跑就行，代码会自动把日志、模型、loss 存到本地文件夹。

## 指定单张 3090 运行，服务器多卡必用，指定只用 0 号 3090：
bash
运行
CUDA_VISIBLE_DEVICES=0 python train.py
整合完整版：
bash
运行
CUDA_VISIBLE_DEVICES=0 python train.py > train_log.txt 2>&1

## 极简全套流程bash运行
1. 创建：screen screen -S train
2. 激活环境：conda activate 你的env
3. 卡0训练+存日志：CUDA_VISIBLE_DEVICES=0 python train.py > train.log 2>&1
4. 脱离后台：Ctrl+A 松开 D
5. 重连恢复：screen -r train


# References
1. [MiniMind项目地址：https://github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)
2. [MiniMind教学视频：https://www.bilibili.com/video/BV1dSwJzqE57](https://www.bilibili.com/video/BV1dSwJzqE57)
3. [GPT1 paper：Improving Language Understanding by Generative Pre-Training](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)
4. [GPT2 paper：Language Models are Unsupervised Multitask Learners](https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-models.pdf)
5. [「AI 第六季」100 刀预算，打造 LLM（附 AI 工程基础）：https://www.bilibili.com/video/BV1R7oVBqERx](https://www.bilibili.com/video/BV1R7oVBqERx)
6. [nanochat项目地址（The best ChatGPT that $100 can buy）：https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)
7. [CS336：https://cs336.stanford.edu/](https://cs336.stanford.edu/)

# Cite

```bibtex
@misc{From Zero to Build LLM,
  author = {Peng Ao},
  title = {From Zero to Build LLMy},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/qipiao/From-Zero-to-Build-LLM}
}
```

# License
MIT