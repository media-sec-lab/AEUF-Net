import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import lib.config.config as cfg
from lib.datasets import roidb as rdl_roidb
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.b1_fuse_1cbam_mask_1 import resnetv3
from lib.utils.timer import Timer
import xlwt,xlrd,os
from xlutils.copy import copy
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if True:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("gt")
        print('Set proposal method: {:s}'.format("gt"))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


class Train:
    def __init__(self):

        # Create network
        if cfg.FLAGS.network == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        elif cfg.FLAGS.network == 'resnet_v1':
            self.net = resnetv3(batch_size=1, num_layers=101)
            # self.net = resnetv1(batch_size=1, num_layers=101)
        else:
            raise NotImplementedError

        self.tfmodel = r'E:\Server_backup\model\b1_fuse_1cbam\b1_fuse_1cbam_128_07_n110/resnet101_faster_rcnn_iter_105000.ckpt'

        output_dir = 'b1_fuse_1cbam_mask_casia_4_07_256_new'

        # self.imdb, self.roidb = combined_roidb("coco_train_filter_single")
        # self.imdb, self.roidb = combined_roidb("dist_NIST_train_new_6")
        self.imdb, self.roidb = combined_roidb("casia_train_all_single")
        # self.imdb, self.roidb = combined_roidb("coverage_train_single")
        # self.imdb, self.roidb = combined_roidb("Nist16_train_all_single")


        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        # self.output_dir = cfg.get_output_dir(self.imdb, 'v12_0.3_momentum_0.001_40k_7')
        # output_dir = 'res_align_rpnsam_b1_3_c_xin7'
        self.output_dir = cfg.get_output_dir(self.imdb, output_dir)
        self.minloss = 100
        self.loss_excel_path = os.path.join(self.output_dir, output_dir+'.xls')
    def train(self):

        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)  # allow_soft_placement = true : select GPU automatically
        tfconfig.gpu_options.allow_growth = True
        # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.10
        sess = tf.Session(config=tfconfig)

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag='default')
            loss = layers['total_loss']
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)



            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            momentum = cfg.FLAGS.momentum
            optimizer = tf.train.MomentumOptimizer(lr, momentum)

            gvs = optimizer.compute_gradients(loss)

            # Double bias
            # Double the gradient of the bias if set
            if cfg.FLAGS.double_bias:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                for i, (g, v) in enumerate(final_gvs):
                    if g is not None:
                        final_gvs[i] = (tf.clip_by_norm(g, 10), v)
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            # writer = tf.summary.FileWriter('default/', sess.graph)
            # valwriter = tf.summary.FileWriter(self.tbvaldir)

        # Load weights
        # Fresh train directly from ImageNet weights
        # print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
        print('Loading initial model weights from {:s}'.format(self.tfmodel))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        # var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)


        var_keep_dic = self.get_variables_in_checkpoint_file(self.tfmodel)
        # Get the variables to restore, ignorizing the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        # restorer.restore(sess, cfg.FLAGS.pretrained_model)
        restorer.restore(sess, self.tfmodel)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        # self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
        self.net.fix_variables(sess, self.tfmodel)
        print('Fixed.')

        
        sess.run(tf.assign(lr, cfg.FLAGS.learning_rate))
        last_snapshot_iter = 0

        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()

        # global_steps = cfg.FLAGS.max_iters
        # decay_steps = 10000
        # decay_rate = 0.96
        # global_step = tf.Variable(tf.constant(0))
        # lr = tf.train.exponential_decay(cfg.FLAGS .learning_rate,global_step,decay_steps,decay_rate, staircase=True)
        loss_total = 0
        loss_rpnbox = 0
        loss_rpncls = 0
        loss_box1 = 0
        loss_cls1 = 0
        loss_mask = 0
        print('START TRAINING: ...')
        while iter < cfg.FLAGS.max_iters + 1:
            # Learning rate
            # if iter == cfg.FLAGS.step_size + 1:
            #     # Add snapshot here before reducing the learning rate
            #     # self.snapshot(sess, iter)
            #     sess.run(tf.assign(lr, cfg.FLAGS .learning_rate * cfg.FLAGS.gamma))
            # if iter%5000 == 0:
            #     sess.run(tf.assign(lr, cfg.FLAGS .learning_rate * 0.85))
            # if iter == 30000 + 1:#nist_new 60k
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 50000 + 1:#nist_new 60k
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 4000 + 1:#coverage
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 30000 + 1:#nist 80k
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 60000 + 1:#nist 80k
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            if iter == 40000 + 1:#casia
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            if iter == 90000 + 1:#casia
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 5000 + 1:#coverage_new
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # if iter == 130000 + 1:
            #     sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * 0.1))
            # learing_rate1 = tf.train.exponential_decay(
            #     learning_rate=0.5, global_step=num_epoch, decay_steps=10, decay_rate=0.9, staircase=True)
            if iter == 1:
                # 创建一个workbook 设置编码
                workbook = xlwt.Workbook(encoding = 'utf-8')
                # 创建一个worksheet
                worksheet = workbook.add_sheet('My Worksheet')
                worksheet.write(0,0, label = 'loss_rpncls')
                worksheet.write(0,1, label = 'loss_rpnbox')
                worksheet.write(0,2, label = 'loss_cls')
                worksheet.write(0,3, label = 'loss_box')
                worksheet.write(0,4, label = 'loss_mask')
                worksheet.write(0,5, label = 'loss_total')
                worksheet.write(0,6, label = 'iteration')

                # 保存
                workbook.save(self.loss_excel_path)
            timer.tic()
            # Get training data, one batch at a time
            blobs = self.data_layer.forward()
            # print(1,blobs['data'].shape)
            # print(2,blobs['gt_boxes'])
            # print(la)
            iter += 1
            # Compute the graph without summary
            # if iter % 100 == 0:
            #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = self.net.train_step_with_summary(
            #         sess, blobs, train_op)
            #     timer.toc()
            #     loss_total = loss_total + total_loss
            #     loss_rpnbox = loss_rpnbox + rpn_loss_box
            #     loss_rpncls = loss_rpncls + rpn_loss_cls
            #     loss_cls1 = loss_cls1 + loss_cls
            #     loss_box1 = loss_box1 + loss_box
            #     run_metadata = tf.RunMetadata()
            #     writer.add_run_metadata(run_metadata, 'step%03d' % iter)
            #     writer.add_summary(summary, iter)
            # else:
            if cfg.FLAGS.USE_MASK is True:
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask, total_loss = \
                    self.net.train_step_with_mask(sess, blobs, train_op)
            else:
                rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(
                sess, blobs, train_op)
            loss_total = loss_total + total_loss
            loss_rpnbox = loss_rpnbox + rpn_loss_box
            loss_rpncls = loss_rpncls + rpn_loss_cls
            loss_cls1 = loss_cls1 + loss_cls
            loss_box1 = loss_box1 + loss_box
            timer.toc()

            # Display training information
            if iter % (cfg.FLAGS.display) == 0:
                if cfg.FLAGS.USE_MASK is True:

                    print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                          '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> loss_mask: %.6f\n ' % \
                          (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_mask,
                           ))
                    print('speed: {:.3f}s / iter'.format(timer.average_time))
                    print('remaining time: {:.3f}h\n'.format(((cfg.FLAGS.max_iters - iter) * timer.average_time) / 3600))
                else:
                    print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                          '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                          (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                    print('speed: {:.3f}s / iter'.format(timer.average_time))
                    print('remaining time: {:.3f}h'.format(((cfg.FLAGS.max_iters - iter) * timer.average_time) / 3600))
                    # print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                    #       '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                    #       (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                    # print('speed: {:.3f}s / iter'.format(timer.average_time))



            # if total_loss<self.minloss and iter>10000:
            #     self.minloss = total_loss
            #     self.snapshot(sess, iter,total_loss,best=True)

            if iter % cfg.FLAGS.snapshot_iterations == 0:
                self.snapshot(sess, iter,total_loss)
                wb = xlrd.open_workbook(self.loss_excel_path)
                newb = copy(wb)
                tabsheet = newb.get_sheet('My Worksheet')
                k = len(tabsheet.rows)# k表示该sheet的最后一行
                tabsheet.write(k, 0, loss_rpncls/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 1, loss_rpnbox/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 2, loss_cls1/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 3, loss_box1/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 4, loss_mask/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 5, loss_total/cfg.FLAGS.snapshot_iterations)
                tabsheet.write(k, 6, iter)
                newb.save(self.loss_excel_path)
                loss_total = 0
                loss_rpnbox = 0
                loss_rpncls = 0
                loss_box1 = 0
                loss_cls1 = 0
                loss_mask = 0

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def snapshot(self, sess, iter,total_loss,best = False):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if best:
            ouput_best = os.path.join(self.output_dir, 'best')
            if not os.path.exists(ouput_best):
                os.makedirs(ouput_best)
            file1 = os.path.join(ouput_best,'minloss.txt')
            with open(file1, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                f.write("total_loss:{:f},iter:{:d}\n".format(total_loss,iter))
            # Store the model snapshot
            filename = 'resnet101_faster_rcnn_best.ckpt'
            filename = os.path.join(ouput_best, filename)
            self.saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename))

            # Also store some meta information, random state, etc.
            nfilename = 'resnet101_faster_rcnn_best.pkl'
            nfilename = os.path.join(ouput_best, nfilename)
            # current state of numpy random
            st0 = np.random.get_state()
            # current position in the database
            cur = self.data_layer._cur
            # current shuffled indeces of the database
            perm = self.data_layer._perm

            # Dump the meta info
            with open(nfilename, 'wb') as fid:
                pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

            return filename, nfilename
        else:
            # Store the model snapshot
            filename = 'resnet101_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
            filename = os.path.join(self.output_dir, filename)
            self.saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename))

            # Also store some meta information, random state, etc.
            nfilename = 'resnet101_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
            nfilename = os.path.join(self.output_dir, nfilename)
            # current state of numpy random
            st0 = np.random.get_state()
            # current position in the database
            cur = self.data_layer._cur
            # current shuffled indeces of the database
            perm = self.data_layer._perm

            # Dump the meta info
            with open(nfilename, 'wb') as fid:
                pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
                pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

            return filename, nfilename


if __name__ == '__main__':
    train = Train()
    train.train()
