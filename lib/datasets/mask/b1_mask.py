# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
# from lib.nets import resnet_thunder_3 as resnet_v1
import numpy as np

from lib.nets.network_mask import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from lib.config import config as cfg
from lib.utils.compact_bilinear_pooling import compact_bilinear_pooling_layer
from lib.nets.roi_align import roi_align
from lib.nets.CBAM import cbam_block_parallel_3
from lib.nets.eca_tf_1 import eca_layer_rpn_1a
def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.FLAGS.weight_decay,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
#    'trainable': cfg.RESNET.BN_TRAIN,
    'trainable': False,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv3(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers
    #self._decide_blocks()

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.FLAGS.max_pool:
        pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.FLAGS.roi_pooling_size, cfg.FLAGS.roi_pooling_size],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.FLAGS.initializer == "truncated":
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      #blocks = [
        #resnet_utils.Block('block1', bottleneck,
        #                   [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        #resnet_utils.Block('block2', bottleneck,
        #                   [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        #resnet_utils.Block('block3', bottleneck,
        #                   [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
        #resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      #]
      blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    elif self._num_layers == 101:
      #blocks = [
        #resnet_utils.Block('block1', bottleneck,
        #                   [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        #resnet_utils.Block('block2', bottleneck,
        #                   [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        #resnet_utils.Block('block3', bottleneck,
        #                   [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        #resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      #]
      blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    elif self._num_layers == 152:
      #blocks = [
        #resnet_utils.Block('block1', bottleneck,
        #                   [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        #resnet_utils.Block('block2', bottleneck,
        #                   [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        #resnet_utils.Block('block3', bottleneck,
        #                   [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        #resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      #]
      blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    assert (0 <= cfg.FLAGS.fixed_blocks < 4)
    if cfg.FLAGS.fixed_blocks == 3:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.FLAGS.fixed_blocks],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    elif cfg.FLAGS.fixed_blocks > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.FLAGS.fixed_blocks],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.FLAGS.fixed_blocks:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.FLAGS.fixed_blocks == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)


    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()  #得到所有可能的archors在原始图像中的坐标（可能超出图像边界）及archors的数量

      # rpn
      net_conv4 = cbam_block_parallel_3(net_conv4, scope='rgb_cbam')
      rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')# shape=(1, ?, ?, 24)
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')#shape=(1, ?, ?, 2)
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")  #训练时得到2000个rois
        #  这里的rpn_bbox_pred是tx，ty，tw, th
        #  现在的rois输出的是原图的坐标
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        #分别得到rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights,
        #储存在self._anchor_targets中，分别应用于后续构造RPN的分类和回归loss。
        #  这里的rpn_bbox_targets:这个是真实的每个anchor与其覆盖最大的ground-truth来计算得到的tx, ty, tw, th。
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")  #nms后得到64个rois
        #现在的rois输出的是原图的坐标
      else:
        if cfg.FLAGS.test_mode == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.FLAGS.test_mode == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError


      #现在的rois输出的是原图的坐标
      box_ind, bbox = self._normalize_bbox(net_conv4, rois, name='rois2bbox')
      # rcnn
      if cfg.FLAGS.pooling_mode == 'crop':
        # pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
        pool5 = roi_align(net_conv4, bbox,
                box_indices=tf.cast(box_ind, tf.int32),
                output_size=[7,7],
                sample_ratio=2)#shape=(64, 7, 7, 1024)
      else:
        raise NotImplementedError

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
    filters = np.einsum('klij->ijlk', filters)
    filters = filters.flatten()
    initializer_srm = tf.constant_initializer(filters)
    if True:
      def truncate_2(x):
        neg = ((x + 2) + abs(x + 2)) / 2 - 2
        return -(2 - neg + abs(2 - neg)) / 2 + 2
      with tf.variable_scope('noise'):
        #kernel = tf.get_variable('weights',
                              #shape=[5, 5, 3, 3],
                              #initializer=tf.constant_initializer(c))
        # conv = tf.nn.conv2d(self.noise, Wcnn, [1, 1, 1, 1], padding='SAME',name='srm')
        conv = slim.conv2d(self._image, 3, [5, 5], trainable=False, weights_initializer=initializer_srm,
                          activation_fn=None, padding='SAME', stride=1, scope='srm')
        conv = truncate_2(conv)
      self._layers['noise']=conv
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        #srm_conv = tf.nn.tanh(conv, name='tanh')
        noise_net = resnet_utils.conv2d_same(conv, 64, 7, stride=2, scope='conv1')
        noise_net = tf.pad(noise_net, [[0, 0], [1, 1], [1, 1], [0, 0]])
        noise_net = slim.max_pool2d(noise_net, [3, 3], stride=2, padding='VALID', scope='pool1')
        #net_sum=tf.concat(3,[net_conv4,noise_net])
        noise_conv4, _ = resnet_v1.resnet_v1(noise_net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope='noise')
        noise_conv4 = cbam_block_parallel_3(noise_conv4, scope='noise_cbam')
        noise_conv4 = eca_layer_rpn_1a(rpn, noise_conv4, 'noise_eca')
    if True:
      box_ind_noise, bbox_noise = self._normalize_bbox(noise_conv4, rois, name='rois2bbox_noise')
      #rfcn roi_align layer
      noise_pool5 = roi_align(noise_conv4, bbox_noise,
                box_indices=tf.cast(box_ind_noise, tf.int32),
                output_size=[7,7],
                sample_ratio=2)
      print(noise_conv4)
      print(noise_pool5)
      # noise_pool5 = self._crop_pool_layer(noise_conv4, rois, "noise_pool5")
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        noise_fc7, _ = resnet_v1.resnet_v1(noise_pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope='noise')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # print(noise_conv4)
      # print(1,fc7)
      # print(2,noise_fc7)
      # print(la)
      bilinear_pool=compact_bilinear_pooling_layer(fc7,noise_fc7,2048)
      # print(bilinear_pool.shape)
      # fc7=tf.Print(fc7,[tf.shape(fc7)],message='Value of %s' % 'fc', summarize=4, first_n=1)
      bilinear_pool = slim.flatten(bilinear_pool, scope='cbp_flatten')
      bilinear_pool=tf.multiply(tf.sign(bilinear_pool),tf.sqrt(tf.abs(bilinear_pool)+1e-12))
      bilinear_pool=tf.nn.l2_normalize(bilinear_pool,dim=1)#shape=(64, 100352)
      # print(1,bilinear_pool.shape)
      # fc6_cbp = slim.fully_connected(bilinear_pool, 2048, scope='fc6')
      # if is_training:
      #     # fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      #     fc6_cbp = slim.dropout(fc6_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout6')
      #
      # # fc7 = slim.fully_connected(fc6, 4096, scope='bbox_fc7')
      # fc7_cbp = slim.fully_connected(fc6_cbp, 2048, scope='fc7')
      # if is_training:
      #     # fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
      #     fc7_cbp = slim.dropout(fc7_cbp, keep_prob=0.5, is_training=True, scope='cbp_dropout7')

      noise_cls_score = slim.fully_connected( bilinear_pool, self._num_classes, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(noise_cls_score, "cls_prob")#shape=(64, 2)

      # print(2,noise_cls_score)
      # print(3,cls_prob)
      # print(la)
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])#shape=(64, 2048)
      # cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
      #                                  trainable=is_training, activation_fn=None, scope='cls_score')
      # cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')#shape=(64, 8)
    if cfg.FLAGS.USE_MASK is True:
      with tf.variable_scope('feature_fuse', 'feature_fuse'):
            # mask_fuse = C_3 * 0.5 + rpn * 0.5
            mask_fuse = net_conv4

            feature_fuse = slim.conv2d(mask_fuse, 1024, [1, 1], padding='VALID', trainable=is_training,
                                       weights_initializer=initializer, scope='mask_fuse')
      mask_box, indices = self._proposal_mask_layer(cls_prob, bbox_pred, rois, 'mask_proposal')
      mask_pool5 = self._crop_pool_layer(feature_fuse, mask_box, "mask_pool5")

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
          mask_fc7, _ = resnet_v1.resnet_v1(mask_pool5,
                                            blocks[-1:],
                                            global_pool=False,
                                            include_root_block=False,
                                            scope='mask_conv')

      self._act_summaries.append(mask_fc7)

      with tf.variable_scope('mask_predict', 'mask_predict'):

          upsampled_features=slim.conv2d_transpose(mask_fc7,256,2,2,activation_fn=None)
          self._act_summaries.append(upsampled_features)
          upsampled_features = slim.conv2d(upsampled_features, 64, [1, 1], normalizer_fn=slim.batch_norm, activation_fn=None,padding='VALID')
          self._act_summaries.append(upsampled_features)
          upsampled_features = slim.batch_norm(upsampled_features, activation_fn=None)
          self._act_summaries.append(upsampled_features)
          upsampled_features = tf.nn.relu(upsampled_features)
          self._act_summaries.append(upsampled_features)

          mask_predictions = slim.conv2d(upsampled_features, num_outputs=2,activation_fn=None,
                                         kernel_size=[1, 1],padding='VALID')
          self._act_summaries.append(mask_predictions)

      self._predictions["mask_out"] = tf.expand_dims(mask_predictions[:, :, :, 1], 3)
      mask_softmax=tf.nn.softmax(mask_predictions)

      self._predictions["mask_softmaxbg"] = tf.expand_dims(mask_softmax[:, :, :, 0], 3)
      self._predictions["mask_softmaxfg"] = tf.expand_dims(mask_softmax[:, :, :, 1], 3)

      self._predictions["rpn_cls_score"] = rpn_cls_score
      self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
      self._predictions["rpn_cls_prob"] = rpn_cls_prob
      self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
      self._predictions["cls_score"] = cls_score
      self._predictions["cls_prob"] = cls_prob
      self._predictions["bbox_pred"] = bbox_pred
      self._predictions["rois"] = rois
      self._predictions["mask_pred"] = mask_predictions

      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred, mask_predictions
    else:
        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = noise_cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois

        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      elif v.name.split('/')[0]=='noise_pred':
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))

  def _normalize_bbox(self, bottom, bbox, name):  #bottom=noise_conv4, bbox=rois
      with tf.variable_scope(name_or_scope=name):
          bottom_shape = tf.shape(bottom)
          height = (tf.to_float(bottom_shape[1]) - 1.)*self._feat_stride[0]
          width = (tf.to_float(bottom_shape[2]) - 1)*self._feat_stride[0]

          indexes, x1, y1, x2, y2 = tf.unstack(bbox, axis=1)
          x1 = x1 / width
          y1 = y1 / height
          x2 = x2 / width
          y2 = y2 / height
          # bboxes = tf.stack([y1, x1, y2, x2], axis=1)
          bboxes = tf.stop_gradient(tf.stack([y1, x1, y2, x2], 1))
          return indexes, bboxes