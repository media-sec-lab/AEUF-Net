# AEUF-Net

This is the implementation of the paper [Image Tampering Localization Using Unified Two-Stream Features Enhanced with Channel and Spatial Attention] (PRCV 2021).

## Usage:
### Train

Perform the training process by using [**train_mask.py**](train_mask.py), where [self.tfmodel](train_mask.py#L74) is the path to a pre-trained model and the function [combined_roidb](train_mask.py#L80) accepts a dataset name as its parameter. Modify the corresponding parts and then run:
```
python train_mask.py
```

### Test

Perform the training process by using [**test_mask.py**](test_mask.py). Change [the path to the model](test_mask.py#L54) and [the name of dataset](test_mask.py#L65) by modifying the defaults and then run:
```
python test_mask.py
```


### Other configurations

[lib/datasets](lib/datasets)：contain the code for different datasets

[lib/datasets/factory.py](lib/datasets/factory.py)：set the path for different datasets

[lib/nets](lib/nets)：contain the code for different networks

[lib/config/config.py](lib/config/config.py)：set the hyper parameters

>'learning_rate'：the learning rate

>'MASK_BATCH'：the number of RoIs in the Mask branch

>'max_iters'：max iterations

>'display'：the number of iterations that the value of loss will be shown

>'snapshot_iterations'：the number of iterations that the model will be saved


## Acknowledgments
The codes are modified from https://github.com/LarryJiang134/Image_manipulation_detection and https://github.com/HuizhouLi/Constrained-R-CNN.
