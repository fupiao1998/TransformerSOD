## 代码组织
论文中所有的参数均保存在config.py中。修改config.py中的31-38行，将SOD和COD数据集的路径修改
## 训练方式
训练任务的选择通过命令行传入。

需要训练COD的话，运行`python train.py --task COD`

需要训练SOD的话，运行`python train.py --task SOD`

训练过程中，checkpoints，log，中间过程可视化均会存储到一个单独的文件夹中，该文件夹在experiments路径下，命名方式为config.py的第34行。

其中log_info表示本次实验的名字，训练之前(非常！)建议手动修改这个名字，为本次实验赋一个有意义的实验名称，默认为`log_info = 'baseline'`
## 测试方式
保证`log_info`的正确性

修改config.py中第50行`param['checkpoint']`对应的checkpoint的路径，选择希望做eval的模型，运行`python test.py --task COD`或者`python test.py --task SOD`即可