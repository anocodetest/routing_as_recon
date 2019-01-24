import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import model
from config import cfg
import nn_utils as nnu


class Routing_As_Recon(model.Model):
    def inference(self, features, reuse, is_train):
        batch_size = cfg.batch_size
        image_size = features['height']
        image_depth = features['depth']
        data_size = image_size
        image = features['images']
        num_classes = features['num_classes']

        initializer_kernel = tf.truncated_normal_initializer(stddev=1e-2, dtype=tf.float32)
        initializer_bias = tf.constant_initializer(0.1)

        with tf.variable_scope('conv1', reuse=reuse) as scope:
            output = slim.conv2d(image, num_outputs=cfg.conv1_channel_num, kernel_size=[9, 9], stride=1,
                                 padding=cfg.padding, weights_initializer=initializer_kernel, biases_initializer=initializer_bias,
                                 trainable=is_train)

        if cfg.padding == 'VALID':
            data_size = data_size - 8
        else:
            data_size = data_size

        with tf.variable_scope('conv2', reuse=reuse) as scope:
            with tf.variable_scope('pose', reuse=reuse) as scope:
                pose = slim.conv2d(output, num_outputs=16*cfg.num_capsule_primary, kernel_size=[9, 9], stride=2,
                                     padding=cfg.padding, weights_initializer=initializer_kernel, activation_fn=None,
                                     biases_initializer=None, trainable=is_train)

        if cfg.padding == 'VALID':
            data_size = int(np.floor((data_size - 8) / 2))
        else:
            data_size = int(data_size / 2)

        with tf.variable_scope('digit_caps', reuse=reuse)as scope:
            output = tf.reshape(pose, [batch_size, -1, 16])
            output = nnu.squash_with_bias(output, is_train)

            with tf.variable_scope('transform', reuse=reuse) as scope1:
                u_hats = nnu.mat_transform(output, num_classes, is_train)
                u_hats = u_hats/tf.norm(u_hats, axis=-1, keep_dims=True)

            with tf.variable_scope('routing', reuse=reuse) as scope1:
                output, logits = nnu.routing_as_recon(u_hats, is_train)

        if cfg.remake:
            remakes = self.remake_evaluate(output, features, reuse=reuse, is_train=is_train)
        else:
            remakes = None

        return model.Inferred(logits, remakes)