# From-Zero-to-Build-LLM

从零开始手撕预训练一个大语言模型，包括简单的复现minimind，进一步训练1B级别的模型


# 复现MiniMind

## 服务器配置
8卡NVIDIA GeForce RTX 3090

NVIDIA-SMI 535.154.05

Driver Version: 535.154.05

CUDA Version: 12.2 

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

## 



# Screen训练+日志log
1. 新建一个后台窗口（训练专用）
bash运行`screen -S train3090`，
train3090 自定义名字，方便区分
2. 进入窗口后，正常操作 bash运行

cd /你的github项目路径

conda activate 你的环境名

python train.py

五、结合 训练日志 + screen 最佳组合
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

六、指定单张 3090 运行（必加，避免抢卡）
服务器多卡必用，指定只用 0 号 3090：
bash
运行
CUDA_VISIBLE_DEVICES=0 python train.py
整合完整版：
bash
运行
CUDA_VISIBLE_DEVICES=0 python train.py > train_log.txt 2>&1

七、极简全套流程（你以后固定这么用）
bash
运行
# 1. 创建screen
screen -S train

# 2. 激活环境
conda activate 你的env

# 3. 卡0训练+存日志
CUDA_VISIBLE_DEVICES=0 python train.py > train.log 2>&1

# 4. 脱离后台：Ctrl+A 松开 D
# 5. 重连恢复：screen -r train
八、screen 和 tmux 简单对比
screen：命令少、上手快、服务器默认自带 → 最推荐你用
tmux：功能更强、分屏好看、快捷键多 → 适合长期运维
九、常见坑
一定要先 conda activate 再跑代码，不然用不到 GPU
不要直接关终端，先 Ctrl+A+D 脱离
想彻底停止训练，进窗口 Ctrl+C 再 exit
需要我给你一条复制即用的终极启动命令（指定 3090+conda + 后台 + 日志全包）吗？
