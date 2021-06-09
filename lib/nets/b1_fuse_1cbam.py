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

from lib.nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from lib.config import config as cfg
# from lib.utils.compact_bilinear_pooling import compact_bilinear_pooling_layer
# from lib.nets.roi_align import roi_align
# from lib.nets.CBAM import cbam_block_parallel_3
# from lib.nets.eca_tf_1 import eca_layer_rpn_1a
def channel_attention_module(inputs, reduction_ratio, reuse=None, scope='channel_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            input_channel = inputs.get_shape().as_list()[-1]
            num_squeeze = input_channel // reduction_ratio

            avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)
            avg_pool = slim.fully_connected(avg_pool, num_squeeze, activation_fn=None, reuse=None, scope='fc1')
            avg_pool = slim.fully_connected(avg_pool, input_channel, activation_fn=None, reuse=None, scope='fc2')
            assert avg_pool.get_shape()[1:] == (1, 1, input_channel)

            max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)
            max_pool = slim.fully_connected(max_pool, num_squeeze, activation_fn=None, reuse=True, scope='fc1')
            max_pool = slim.fully_connected(max_pool, input_channel, activation_fn=None, reuse=True, scope='fc2')
            assert max_pool.get_shape()[1:] == (1, 1, input_channel)

            scale = tf.nn.sigmoid(avg_pool + max_pool)
            # return scale
            channel_attention = scale * inputs

            return channel_attention


def spatial_attention_module(inputs, kernel_size=7, reuse=None, scope='spatial_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
            assert max_pool.get_shape()[-1] == 1

            concat = tf.concat([avg_pool, max_pool], axis=3)
            assert concat.get_shape()[-1] == 2

            concat = slim.conv2d(concat, 1, kernel_size, padding='SAME', activation_fn=None, scope='conv')
            scale = tf.nn.sigmoid(concat)
            # return scale

            spatial_attention = scale * inputs

            return spatial_attention

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

  # def _crop_pool_layer(self, bottom, rois, name):
  #   with tf.variable_scope(name) as scope:
  #     batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
  #     # Get the normalized coordinates of bboxes
  #     bottom_shape = tf.shape(bottom)
  #     height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
  #     width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
  #     x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
  #     y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
  #     x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
  #     y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
  #     # Won't be backpropagated to rois anyway, but to save time
  #     bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
  #     if cfg.FLAGS.max_pool:
  #       pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
  #       crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
  #                                        name="crops")
  #       crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
  #     else:
  #       crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.FLAGS.roi_pooling_size, cfg.FLAGS.roi_pooling_size],
  #                                        name="crops")
  #   return crops
  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
        batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # Get the normalized coordinates of bboxes
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / np.float32(self._feat_stride[0])
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / np.float32(self._feat_stride[0])
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / np.float32(self._feat_stride[0])
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / np.float32(self._feat_stride[0])
        # Won't be back-propagated to rois anyway, but to save time

        pre_pool_size = 7#cfg.POOLING_SIZE
        spacing_w = (x2 - x1) / pre_pool_size
        spacing_h = (y2 - y1) / pre_pool_size
        x1 = (x1 + spacing_w / 2) / (tf.to_float(bottom_shape[2]) - 1.)
        y1 = (y1 + spacing_h / 2) / (tf.to_float(bottom_shape[1]) - 1.)
        nw = spacing_w * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[2]) - 1.)
        nh = spacing_h * tf.to_float(pre_pool_size - 1) / (tf.to_float(bottom_shape[1]) - 1.)
        x2 = x1 + nw
        y2 = y1 + nh

        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],name="crops")
        return crops

  def PAM_module_1(self,inputs,n_inputs, name=''):
      #https://blog.csdn.net/xh_hit/article/details/88575853
      with tf.variable_scope("PAM_" + name, reuse=None):
          # gamma  = Layer.add_weight(shape=(1,),
          #             initializer=tf.zeros_initializer(),
          #             name='gamma',
          #             regularizer=None,
          #             trainable=is_training,
          #             constraint=None)

          # batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]
          input_shape1 = inputs.get_shape().as_list()
          _, _, _, filters = input_shape1

          input_shape = tf.shape(inputs)
          _, h, w, _ = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

          B = slim.conv2d(inputs, filters // 8, [1, 1], padding="SAME",activation_fn=None,scope='B_layer')
          C = slim.conv2d(inputs, filters // 8, [1, 1], padding="SAME",activation_fn=None,scope='C_layer')
          D = slim.conv2d(inputs, filters, [1, 1], padding="SAME",activation_fn=None,scope='D_layer')

          vec_b = tf.reshape(B, [-1, h * w, filters // 8])
          vec_cT = tf.transpose(tf.reshape(C, [-1, h * w, filters // 8]), [0, 2, 1])
          bcT = tf.matmul(vec_b, vec_cT)
          softmax_bcT = tf.nn.softmax(bcT)

          nB = slim.conv2d(n_inputs, filters // 8, [1, 1], padding="SAME",activation_fn=None,scope='nB_layer')
          nC = slim.conv2d(n_inputs, filters // 8, [1, 1], padding="SAME",activation_fn=None,scope='nC_layer')
          nD = slim.conv2d(n_inputs, filters, [1, 1], padding="SAME",activation_fn=None,scope='nD_layer')

          vec_nb = tf.reshape(nB, [-1, h * w, filters // 8])
          vec_ncT = tf.transpose(tf.reshape(nC, [-1, h * w, filters // 8]), [0, 2, 1])
          nbcT = tf.matmul(vec_nb, vec_ncT)
          softmax_nbcT = tf.nn.softmax(nbcT)

          vec_d = tf.reshape(D, [-1, h * w, filters])
          bcTd = tf.matmul(softmax_bcT, vec_d)

          vec_nd = tf.reshape(nD, [-1, h * w, filters])
          nbcTd = tf.matmul(softmax_nbcT, vec_nd)

          bcTd = tf.reshape(bcTd+nbcTd, [-1, h, w, filters])

          # out =  bcTd + inputs
          out = bcTd
          return out

  def CAM_module_1(self,inputs,n_inputs,name=''):
      #https://blog.csdn.net/xh_hit/article/details/88575853
      #https://github.com/niecongchong/DANet-keras/blob/master/layers/attention.py
      with tf.variable_scope("CAM_" + name, reuse=None):
          # gamma  = Layer.add_weight(shape=(1,),
          #             initializer=tf.zeros_initializer(),
          #             name='gamma',
          #             regularizer=None,
          #             trainable=is_training,
          #             constraint=None)
          input_shape1 = inputs.get_shape().as_list()
          _, _, _, filters = input_shape1

          input_shape = tf.shape(inputs)
          _, h, w, _ = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

          vec_a = tf.reshape(inputs, [-1, h * w, filters])
          vec_aT = tf.transpose(vec_a, [0, 2, 1])
          vec_e = tf.reshape(n_inputs, [-1, h * w, filters])

          aTa = tf.matmul(vec_aT, vec_e)
          softmax_aTa = tf.nn.softmax(aTa)
          aaTa = tf.matmul(vec_a, softmax_aTa)
          aaTa = tf.reshape(aaTa, [-1, h, w, filters])

          # out = aaTa + inputs
          out = aaTa
          return out
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
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.FLAGS.fixed_blocks],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4a, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.FLAGS.fixed_blocks:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
        # net_conv4_pam = self.PAM_module(net_conv4a,name='net_conv4_pam')
        # net_conv4_cam = self.CAM_module(net_conv4a,name='net_conv4_cam')
        #
        # net_conv4 = net_conv4_pam + net_conv4_cam

    self._act_summaries.append(net_conv4a)
    self._layers['head'] = net_conv4a
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
        noise_conv4a, _ = resnet_v1.resnet_v1(noise_net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope='noise')
        net_conv4_pam = spatial_attention_module(net_conv4a, kernel_size=7, reuse=None, scope='rgb_spatial_attention')
        net_conv4_cam = channel_attention_module(net_conv4a, reduction_ratio=16, reuse=None,scope='rgb_channel_attention')

        noise_conv4_pam = spatial_attention_module(noise_conv4a, kernel_size=7, reuse=None, scope='noise_spatial_attention')
        noise_conv4_cam = channel_attention_module(noise_conv4a, reduction_ratio=16, reuse=None,scope='noise_channel_attention')

        conv4_pam = slim.conv2d(net_conv4_pam+noise_conv4_pam, 1024, 1, padding='SAME', activation_fn=None, scope='conv4_pam')
        conv4_cam = slim.conv2d(net_conv4_cam+noise_conv4_cam, 1024, 1, padding='SAME', activation_fn=None, scope='conv4_cam')

        net_conv4 = slim.conv2d(conv4_pam + conv4_cam, 1024, 1, padding='SAME', activation_fn=None, scope='net_conv4')
        #
        # noise_conv4 = noise_conv4_pam + noise_conv4_cam
        # noise_conv4 = cbam_block_parallel_3(noise_conv4, scope='noise_cbam')
        # noise_conv4 = eca_layer_rpn_1a(rpn, noise_conv4, 'noise_eca')

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()  #得到所有可能的archors在原始图像中的坐标（可能超出图像边界）及archors的数量

      # rpn
      # net_conv4 = cbam_block_parallel_3(net_conv4, scope='rgb_cbam')
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
      # box_ind, bbox = self._normalize_bbox(net_conv4, rois, name='rois2bbox')
      # rcnn
      if cfg.FLAGS.pooling_mode == 'crop':
        pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
        # pool5 = roi_align(net_conv4, bbox,
        #         box_indices=tf.cast(box_ind, tf.int32),
        #         output_size=[7,7],
        #         sample_ratio=2)#shape=(64, 7, 7, 1024)
      else:
        raise NotImplementedError
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      noise_cls_score = slim.fully_connected( fc7, self._num_classes, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(noise_cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
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