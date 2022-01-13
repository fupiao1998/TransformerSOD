## 简介
本项目是论文《Generative Transformer for Accurate and Reliable Salient Object Detection》中所有实验对应的开源代码。

## 配置
所有实验在一块3090显卡上完成，所使用的pytorch版本为1.9.1，timm版本为0.4.5。

同时，需要手动下载swin transformer backbone的[模型](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth)，并放置在```model```路径下
## 实验
### 数据集
我们使用的所有数据集均上传在[]中，可以将下载并同时修改```path_config.py```中数据集路径以及对应机器的hostname。获取hostname的方式：
```
import socket
hostname = socket.getfqdn(socket.gethostname(  ))
print(hostname)
```
### config
```config.py```中可以配置所有实验对应的实验配置，超参数等。当前```config.py```中的默认参数即为本文中所有实验对应的最终参数。与此同时，使用者可以通过在命令行中挂载相应的args参数，或者直接在```config.py```修改对应的内容的方式，实现不同任务，模型，参数的组合。args中主要参数的含义表示如下：
* **task**：训练的任务
* **backbone**：使用的backbone模型，可选择swin transformer，resnet50以及vit
* **neck**：实现backbone的channel reduce，可通过simple conv或者rcab模块实现，本文中统一使用的是simple conv
* **decoder**：实现的decoder，本文中统一使用的是'cat'decoder，即将不同尺度的特征简单concat然后实现saliency map的输出
* **fusion**：几种不同的用于RGBD-SOD任务中depth的使用方式
* **uncer_method**：不同的uncertainty模型，其中basic为不使用uncertainty建模
* **log_info**：表示本次实验希望命名的名称，默认为REMOVE

### demo
我们实现了简单的demo用于实现一个RGB输入的saliency map的生成以及网络中间特征的可视化。其中```[ckpt_path]```为想要测试的预训练模型。
```
python demo.py --ckpt [ckpt_path]
```
![alt 特征图可视化](assert/assert.png)
### 训练
通过运行```python train.py```即可根据当前```config.py```中的配置信息，进行默认配置的训练。

若需要训练其他的设置，可在```config.py```中修改对应的配置选项。例如：

```python train.py --task SOD --uncer_method ganabp```

表示训练一个使用IGAN为uncertainty模型，以swin transformer为backbone的SOD任务模型。

### 测试
在设置好配置文件的情况下，直接运行```python test.py --ckpt [ckpt_path]```即可实现saliency map的输出以及对应的MAE的评估。
### 显著性图
为了便于对比，我们提供了各种实验设置下模型输出的saliency map，可通过该[链接]下载。
## BIB
如果您认为我们的代码对您的科研有帮助，请引用：
```
@article{mao2021transformer,
  title={Transformer transforms salient object detection and camouflaged object detection},
  author={Mao, Yuxin and Zhang, Jing and Wan, Zhexiong and Dai, Yuchao and Li, Aixuan and Lv, Yunqiu and Tian, Xinyu and Fan, Deng-Ping and Barnes, Nick},
  journal={arXiv preprint arXiv:2104.10127},
  year={2021}
}
```
## 联系
如果您认为有问题需要讨论，请通过github的issue或我的邮箱与我联系：maoyuxin@mail.nwpu.edu.cn

