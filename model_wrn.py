import tensorflow as tf
import tensorflow.contrib.slim as slim

import model
import nn_utils as nnu
from config import cfg
from model_resnet import Resnet


class Wide_Resnet(Resnet):
    def inference(self, features, reuse, is_train):
        batch_size = cfg.batch_size
        image_size = features['height']
        image_depth = features['depth']
        data_size = image_size
        image = features['images']
        num_classes = features['num_classes']

        with tf.variable_scope('init_conv', reuse=reuse) as scope:
            outputs = nnu.conv2d_clean(image, 16, 3, 1, is_train)

        for i in range(cfg.num_res_block):
            with tf.variable_scope('scale1_block%d' % i, reuse=reuse) as scope:
                outputs = nnu.wide_residual_block(outputs, 16*cfg.widen_factor, 1, is_train)

        with tf.variable_scope('scale2_block0', reuse=reuse) as scope:
            outputs = nnu.wide_residual_block(outputs, 32*cfg.widen_factor, 2, is_train)
        for i in range(1, cfg.num_res_block):
            with tf.variable_scope('scale2_block%d' % i, reuse=reuse) as scope:
                outputs = nnu.wide_residual_block(outputs, 32 * cfg.widen_factor, 1, is_train)

        with tf.variable_scope('scale3_block0', reuse=reuse) as scope:
            outputs = nnu.wide_residual_block(outputs, 64 * cfg.widen_factor, 2, is_train)
        for i in range(1, cfg.num_res_block):
            with tf.variable_scope('scale3_block%d' % i, reuse=reuse) as scope:
                outputs = nnu.wide_residual_block(outputs, 64 * cfg.widen_factor, 1, is_train)

        with tf.variable_scope('fc', reuse=reuse) as scope:
            outputs = nnu.batch_norm_res(outputs, is_train)
            outputs = nnu.relu_ge(outputs)

            outputs = tf.reduce_mean(outputs, axis=[1, 2])
            logits = nnu.fc_res(outputs, num_classes, is_train)

            if not is_train:
                logits = tf.nn.softmax(logits)

        remakes = None

        return model.Inferred(logits, remakes)
