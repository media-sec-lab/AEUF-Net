# EMUNet
  
train_mask.py：训练代码
self.tfmodel：预训练模型存放的路径；
combined_roidb函数里的变量是不同数据集的名称；
修改变量后直接python train_mask.py即可

test_mask.py：测试代码
在'--model'的default中修改模型存放路径；
在'--imdb'的default中修改数据集的名称；
修改变量后直接python test_mask.py即可


lib.datasets：存放不同数据集代码的文件夹
lib.datasets.factory.py：设置不同数据集的路径
lib.nets：存放不同网络代码的文件夹
lib.config.config.py：设置学习率，迭代次数等超参数

config.py里的主要参数有：
'learning_rate'：学习率；
'MASK_BATCH'：Mask分支的RoI数量；
'max_iters'：最大迭代次数；
'display'：每隔多少个迭代就会展示当前的loss；
'snapshot_iterations'：每隔多少个迭代就会保存当前模型；

# Acknowledgments
The codes are modified from https://github.com/LarryJiang134/Image_manipulation_detection and https://github.com/HuizhouLi/Constrained-R-CNN
