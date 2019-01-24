import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import model
from config import cfg
import nn_utils as nnu


class Cnn_Net(model.Model):
    def inference(self, features, reuse, is_train):
        batch_size = cfg.batch_size
        image_size = features['height']
        image_depth = features['depth']
        data_size = image_size
        image = features['images']
        num_classes = features['num_classes']

        initializer_kernel = tf.truncated_normal_initializer(stddev=1e-2, dtype=tf.float32)
        initializer_bias = tf.constant_initializer(0.1)
        regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)

        with tf.variable_scope('conv1', reuse=reuse) as scope:
            output = slim.conv2d(image, num_outputs=cfg.conv1_channel_num, kernel_size=[9, 9], stride=1, padding=cfg.padding,
                                 weights_initializer=initializer_kernel, weights_regularizer=regularizer,
                                 biases_initializer=initializer_bias, biases_regularizer=regularizer,
                                 trainable=is_train)

        if cfg.padding == 'VALID':
            data_size = data_size - 8
        else:
            data_size = data_size

        with tf.variable_scope('conv2', reuse=reuse) as scope:
            output = slim.conv2d(output, num_outputs=16*cfg.num_capsule_primary, kernel_size=[9, 9], stride=2,
                                 padding=cfg.padding, weights_initializer=initializer_kernel,
                                 weights_regularizer=regularizer, biases_initializer=initializer_bias,
                                 biases_regularizer=regularizer, trainable=is_train)


        with tf.variable_scope('fc1', reuse=reuse) as scope:
            output = tf.reshape(output, [batch_size, -1])
            output = slim.fully_connected(output, num_outputs=num_classes, weights_initializer=initializer_kernel,
                                          weights_regularizer=regularizer, biases_initializer=initializer_bias,
                                          biases_regularizer=regularizer, trainable=is_train, activation_fn=None)

        remakes = None

        return model.Inferred(output, remakes)


