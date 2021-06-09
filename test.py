# --------------------------------------------------------
# Tensorflow RGB-N
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou , based on code from Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
from lib.utils.test import test_net
from lib.config import config as cfg
from lib.datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v2 import resnetv1
# from lib.nets.b1_fuse_1cbam import resnetv3
# from lib.nets.b1 import resnetv3
# from lib.nets.res_align_rpnsam_b1 import resnetv3
# from lib.nets.b1_edge_1 import resnetv3
# from lib.nets.b1_fuse_1cbam import resnetv3
from lib.nets.b1_fuse_sc1aAddcfe import resnetv3
# from lib.nets.res_align_are_at_2 import resnetv3
# from lib.nets.res_align_rpnsam_b1_2to1 import resnetv3
# from lib.nets.res_align_are_at_2 import resnetv3
# from lib.nets.res2net_align_1 import resnetv3
# from lib.nets.res_align_3 import resnetv3
# from lib.nets.r_fcn_align import resnetv3
# from nets.vgg16 import vgg16
# from nets.resnet_v1 import resnetv1
# from nets.resnet_v1_noise import resnet_noise
# from nets.resnet_fusion import resnet_fusion
# from nets.resnet_fusion_noise import resnet_fusion_noise
from tensorflow.python import pywrap_tensorflow
import pdb

def get_variables_in_checkpoint_file(file_name):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  # parser.add_argument('--cfg', dest='cfg_file',
  #           help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res101_110k_150k/resnet101_faster_rcnn_iter_55000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res101_150k_xin6/resnet101_faster_rcnn_iter_145000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_rpnsam_b1_3_xin9/resnet101_faster_rcnn_iter_160000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_are_at_2_xin10_drop/resnet101_faster_rcnn_iter_140000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/casia_train_all_single/res_align_are_at_2_xin10_casia_000/resnet101_faster_rcnn_iter_150000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_concat_xin9/resnet101_faster_rcnn_iter_100000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_3_150k_xin6/resnet101_faster_rcnn_iter_135000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_4_xin8/resnet101_faster_rcnn_iter_140000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_rpnsam_b1_4_xin8/resnet101_faster_rcnn_iter_125000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res101_150k_xin6/resnet101_faster_rcnn_iter_100000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_rpnsam_b1_150k_xin6/resnet101_faster_rcnn_iter_150000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_aren_2_nofix_new1/resnet101_faster_rcnn_iter_160000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_srm_1_new1/resnet101_faster_rcnn_iter_160000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_srm_1_nofix_new1/resnet101_faster_rcnn_iter_130000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_rgb_new2/resnet101_faster_rcnn_iter_160000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_are_at_2_two_new1/resnet101_faster_rcnn_iter_165000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_are_at_2_xin10_drop_new1/resnet101_faster_rcnn_iter_165000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_are_at_2_new1_256/resnet101_faster_rcnn_iter_150000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_rpnsam_b1_2to1_new1/resnet101_faster_rcnn_iter_150000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_aren_2_nofix_new1_000/resnet101_faster_rcnn_iter_130000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/b1_256/resnet101_faster_rcnn_iter_150000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/res_align_rpnsam_b1_64_0816/resnet101_faster_rcnn_iter_110000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/b1_64_new_2bias_0817/resnet101_faster_rcnn_iter_115000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/are_net_x_0.7_0917/resnet101_faster_rcnn_iter_180000.ckpt', type=str)
            default='/data/cxm/code/Image_manipulation_detection-master/default/coco_train_filter_single/b1_fuse_sc1aAddcfe_128_07_n110/resnet101_faster_rcnn_iter_100000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/casia_train_all_single/res_align_are_at_2_rpn_new1_casia_000/resnet101_faster_rcnn_iter_60000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coverage_train_all_single/res_align_rpnsam_b_150k_xin4/resnet101_faster_rcnn_iter_20000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/casia_train_all_single/res_align_rpnsam_b1_150k_xin4_casia_000/resnet101_faster_rcnn_iter_135000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/casia_train_all_single/res_align_rpnsam_b1_3_150k_xin4_casia_000/resnet101_faster_rcnn_iter_70000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/Nist16_train_all_single/res_align_rpnsam_b1_150k_xin4_Nist16_000/resnet101_faster_rcnn_iter_60000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coverage_train_all_single/res_align_rpnsam_b1_150k_xin4_coverage__000/resnet101_faster_rcnn_iter_10000.ckpt',type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/casia_train_all_single/res_align_rpnsam_b1_150k_xin4_casia_000/resnet101_faster_rcnn_iter_135000.ckpt',type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/coverage_train_all_single/res_align_rpnsam_b1_150k_xin4_coverage__000/resnet101_faster_rcnn_iter_10000.ckpt',type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/Nist16_train_all_single/res_align_rpnsam_b1_3_xin7_nist_000_1/resnet101_faster_rcnn_iter_80000.ckpt', type=str)
            # default='/data/cxm/code/Image_manipulation_detection-master/default/Nist16_train_all_single/res101_110k_150k_nist_000/resnet101_faster_rcnn_iter_80000.ckpt', type=str)

  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            # default='casia_test_all_single', type=str)
            default='coco_test_filter_single', type=str)
            # default='DIY_dataset', type=str)
            # default='coverage_test_all_single', type=str)
            # default='columbia_test_all_single', type=str)
            # default='Nist16_test_all_single', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      # default='res101', type=str)
                      default='rfcn', type=str)
  # parser.add_argument('--set', dest='set_cfgs',
  #                       help='set config keys', default=None,
  #                       nargs=argparse.REMAINDER)

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  # print(444)
  print('Called with args:')
  print(args)

  # if args.cfg_file is not None:
  #   cfg_from_file(args.cfg_file)
  # if args.set_cfgs is not None:
  #   cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # net = resnetv1(batch_size=1, num_layers=101)
  # load network
  if args.net == 'vgg16':
    net = vgg16(batch_size=1)
    print(args.net)
  elif args.net == 'res101':
    net = resnetv1(batch_size=1, num_layers=101)
  elif args.net == 'rfcn':
    net = resnetv3(batch_size=1, num_layers=101)
  else:
    raise NotImplementedError

  # load model
  # net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
  #                         anchor_scales=cfg.ANCHOR_SCALES,
  #                         anchor_ratios=cfg.ANCHOR_RATIOS)
  # net.create_architecture(sess, "TEST", imdb.num_classes, tag='default', anchor_scales=[8, 16, 32])
  net.create_architecture(sess, "TEST", imdb.num_classes, tag='default')

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #pdb.set_trace()
    #for v in variables:
      #print('Varibles: %s' % v.name)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)

  sess.close()
