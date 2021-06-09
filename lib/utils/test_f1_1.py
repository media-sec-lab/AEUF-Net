# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os

from lib.utils.timer import Timer
# from utils.cython_nms import nms, nms_new
from lib.utils.py_cpu_nms import py_cpu_nms as nms
from lib.utils.blob import im_list_to_blob

# from model.config import cfg, get_output_dir
from lib.config.config import get_output_dir
from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform_inv


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.FLAGS2["pixel_means"]

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.FLAGS2["test_scales"]:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.FLAGS.test_max_size:
            im_scale = float(cfg.FLAGS.test_max_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""
    for i in range(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im):
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    # seems to have height, width, and image scales
    # still not sure about the scale, maybe full image it is 1.
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    _, scores, bbox_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.FLAGS.test_bbox_reg:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.FLAGS.DET_THRESHOLD))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes
def bounding_box(mask):
    # box_list=[]
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_out = np.zeros(gray.shape[:2], dtype=np.float)
    # contours = sorted(contours, key=lambda i: len(i),reverse=True)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        x1=x
        y1=y
        x2=x+w
        y2=y+h
        bbox = [x1,y1,x2,y2]
        mask_out[bbox[1]:bbox[3], bbox[0]:bbox[2]]=1.
    return mask_out

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
    np.random.seed(cfg.FLAGS.rng_seed)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    all_f1 = []
    all_auc = []
    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        # print(imdb.image_path_at(i))
        im = cv2.imread(imdb.image_path_at(i))

        mask_gt = cv2.imread(imdb.mask_path_at(i))
        # mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
        # ret, mask_gt = cv2.threshold(mask_gt, 127, 255, cv2.THRESH_BINARY)
        # mask_gt = (mask_gt / 255.0).astype(np.float32)
        mask_gt = bounding_box(mask_gt).astype(np.float32)

        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im)
        _t['im_detect'].toc()

        #先得到mask，计算f1
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep, :]
            batch_ind = np.where(cls_scores > 0.)[0]
            if batch_ind.shape[0] == 0:
                f1 = 1e-10
                auc_score = 1e-10
            else:
                # print(i+1,'im',np.shape(im),'mask_gt',np.shape(mask_gt),'cls_dets',np.shape(cls_dets))
                auc_score, f1 = f1_detections(im, mask_gt, cls_dets, thresh=0.01)
                # print('auc_score',auc_score,'f1',f1)
            with open('test_filter_single_new1.txt', 'a') as f:  # 设置文件对象
                f.write('%s  %s %.5f %s %.5f\n' % (os.path.basename(imdb.mask_path_at(i)),'auc_score',auc_score,'f1',f1))
            # print('auc_score:{:.3f} f1:{:.3f}'.format(auc_score,f1),end='\r')
        all_f1.append(f1)
        all_auc.append(auc_score)


        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets
        # print('all_boxes',np.shape(all_boxes))
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()
        # print('all_boxes',all_boxes)
        # print(a)
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s remaining time: {:.3f}m' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      # _t['misc'].average_time,((num_images-i-1)*_t['im_detect'].average_time)/60),end="\r")
                      _t['misc'].average_time,((num_images-i-1)*_t['im_detect'].average_time)/60),end="\r")
    print('\n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Test Results:')
    print('Average F1  Score: %.3f' % np.average(all_f1))
    print('Average AUC Score: %.3f' % np.average(all_auc))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    #
    # print('Evaluating detections')
    # imdb.evaluate_detections(all_boxes, output_dir)

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    # assert prediction.dtype == np.uint8
    # assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    y_test=gt.flatten()
    y_pred=prediction.flatten()
    precision,recall,thresholds=metrics.precision_recall_curve(y_test,y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    return precision, recall,auc_score


def cal_fmeasure(precision, recall):

    max_fmeasure = max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])
    return max_fmeasure
def f1_detections(im, mask_gt, dets, thresh=0.01):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= 0.9)[0]   #给出满足条件的数组索引
    # print(len(inds))
    if len(inds)==0:
        inds = np.where(dets[:, -1] >= 0.5)[0]
        if len(inds)==0:
            inds = np.where(dets[:, -1] >= 0.1)[0]
            if len(inds) == 0:
                inds = np.where(dets[:, -1] >= 0.01)[0]
    # if len(inds) == 0:
    #     return

    mask_out = np.zeros(im.shape[:2],dtype=np.float)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # print('bbox',bbox)
        # print('score',score)

        bbox = bbox.astype(int)
        mask_out[bbox[1]:bbox[3], bbox[0]:bbox[2]]=1.

        precision, recall, auc_score = cal_precision_recall_mae(mask_out, mask_gt)
        f1 = cal_fmeasure(precision, recall)
        f1 = np.max(np.array(f1))
    return auc_score, f1