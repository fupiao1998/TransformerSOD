## Introduction
This project is the open source code corresponding to all experiments in the paper "Generative Transformer for Accurate and Reliable Salient Object Detection". Chinese version is at [简体中文](README_CN.md)

##  Configuration
All experiments are done on a 3090 graphics card, the pytorch version used is 1.9.1, and the timm version is 0.4.5. 

At the same time, you need to manually download the [pre-trained model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth) of the swin transformer backbone ，and placed under the ````model```` folder.
## Experiment 
### Dataset
All datasets we use are uploaded in [], you can download and modify the dataset path and the hostname of the corresponding machine in ```path_config.py``` at the same time. How to get hostname: 
```
import socket
hostname = socket.getfqdn(socket.gethostname(  ))
print(hostname)
```
### Config
```config.py``` can configure the experimental configuration, hyperparameters, etc. corresponding to all experiments. The default parameters in the current ``config.py`` are the final parameters corresponding to all experiments in this article. At the same time, users can implement the combination of different tasks, models, and parameters by mounting the corresponding args parameters on the command line, or directly modifying the corresponding content in ```config.py```. The meaning of the main parameters in args is expressed as follows: 
* **task**：training task
* **backbone**：The backbone model used, you can choose swin transformer, resnet50 and vit
* **neck**：Implement the channel reduce of the backbone, which can be implemented through the simple conv or rcab module. The simple conv is used uniformly in this article. 
* **decoder**：The implemented decoder, the 'cat' decoder is used uniformly in this article, that is, the features of different scales are simply concat and then the output of the saliency map is realized 
* **fusion**：Several different ways of using depth for RGBD-SOD tasks 
* **uncer_method**：Different uncertainty models, where basic is modeled without uncertainty 
* **log_info**：Indicates the name that this experiment wants to name, the default is REMOVE 

### Demo
We implement a simple demo for generating an RGB input saliency map and visualizing the intermediate features of the network. where ```[ckpt_path]``` is the pretrained model you want to test. 
```
python demo.py --ckpt [ckpt_path]
```
![alt feater_vis](assert/assert.png)
### Training
By running ```python train.py```, the default configuration can be trained according to the configuration information in the current ```config.py```.

If you need to train other settings, you can modify the corresponding configuration options in ``config.py``. For example:

```python train.py --task SOD --uncer_method ganabp```

The above command means to train a SOD task model that uses IGAN as the uncertainty model and swin transformer as the backbone.

### 测试
With the configuration file set up, run ```python test.py --ckpt [ckpt_path]``` directly to output the saliency map and evaluate the corresponding MAE.
### 显著性图
For ease of comparison, we provide the saliency map output by the model under various experimental settings, which can be downloaded from this [link] 。
## BIB
If you think our code is helpful for your research, please cite:
```
@article{mao2021transformer,
  title={Transformer transforms salient object detection and camouflaged object detection},
  author={Mao, Yuxin and Zhang, Jing and Wan, Zhexiong and Dai, Yuchao and Li, Aixuan and Lv, Yunqiu and Tian, Xinyu and Fan, Deng-Ping and Barnes, Nick},
  journal={arXiv preprint arXiv:2104.10127},
  year={2021}
}
```
## 联系
If you think there is a problem to discuss, please contact me via issue on github or my email :maoyuxin@mail.nwpu.edu.cn.

At the same time, we hope that our framework can include more transformer-based work for SOD tasks, so please stay tuned for our updates. Pulling requests are also welcome!


