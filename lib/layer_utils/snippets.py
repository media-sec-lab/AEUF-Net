# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors

#这个函数的意思大概就是将特征图上的锚返回到原图上
def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'给定不同比例生成锚点的包装函数也返回可变“长度”的锚点数量
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]#anchor的数量，为9
    shift_x = np.arange(0, width) * feat_stride#将特征图的宽度进行16倍延伸至原图，以width=4为例子，则shfit_x=[0,16,32,48]
    shift_y = np.arange(0, height) * feat_stride#将特征图的高度进行16倍衍生至原图
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)#生成原图的网格点
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()#若width=50，height=38，生成（50*38）*4的数组
    # 如 [[0,0,0,0],[16,0,16,0],[32,0,32,0].......]，shift中的前两个坐标和后两个一样（保持右下和左上的坐标一样），是从左到右，从上到下的坐标点（映射到原图）
    #shifts就是对（shift_x, shift_y）进行组合，其中shift_x是对x坐标进行移动，shift_y是对y坐标进行移动，
    # 综合起来就是将基础的中心为（7.5，7.5）的9个anchor平移到全图上，覆盖所有可能的区域。
    K = shifts.shape[0]#k=50*38
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    # 其实意思就是右下角坐标和左上角的左边都加上同一个变换坐标
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)#三维变两维，（50*38*9，4），此处就是将特征层的anchor坐标转到原图上的区域
    length = np.int32(anchors.shape[0])#length=50*38*9
    #上述代码就是完成了9个base anchor 的移动，输出结果就是50*38*9个anchor。那么到此，所有的anchor都生成了，
    # 当然了，所有的anchor也和特征图产生了一一对应的关系了。
    return anchors, length
