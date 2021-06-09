# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes


def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
    """A layer that just selects the top region proposals
       without using non-maximal suppression,
       For details please see the technical report
    """
    rpn_top_n = cfg.FLAGS.rpn_top_n#300
    im_info = im_info[0]

    scores = rpn_cls_prob[:, :, :, num_anchors:]

    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    # 统计有多少个框
    length = scores.shape[0]
    if length < rpn_top_n:
        # Random selection, maybe unnecessary and loses good proposals
        # But such case rarely happens
        top_inds = npr.choice(length, size=rpn_top_n, replace=True)
    else:
        # 从大到小排序，取列索引
        top_inds = scores.argsort(0)[::-1]
        top_inds = top_inds[:rpn_top_n]# 取前大的300个
        top_inds = top_inds.reshape(rpn_top_n, )

    # Do the selection here
    # 选择/重排
    # 按照索引提取anchor数据
    anchors = anchors[top_inds, :]
    rpn_bbox_pred = rpn_bbox_pred[top_inds, :]
    scores = scores[top_inds]

    # Convert anchors into proposals via bbox transformations
    # bbox_transform_inv : 根据anchor和偏移量计算proposals
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)

    # Clip predicted boxes to image
    # clip_boxes : proposals的边界限制在图片内
    proposals = clip_boxes(proposals, im_info[:2])

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    # 和 proposal_layer 一样，多出来一列0，然后拼接
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob, scores
