import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import model
from config import cfg
import nn_utils as nnu


class Resnet(model.Model):
    def __init__(self):
        """Initializes the model parameters.

        Args:
          hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        """
        with tf.device('/cpu:0'):
            self._global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                trainable=False)

            learning_rate = tf.train.exponential_decay(
                learning_rate=cfg.init_learning_rate,
                global_step=self._global_step,
                decay_steps=cfg.decay_steps,
                decay_rate=cfg.exp_decay_rate)
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
            learning_rate_const = tf.constant(cfg.init_learning_rate, dtype=tf.float32)

            if cfg.learning_rate_decrease == 'exp':
                self._learning_rate = tf.maximum(learning_rate, 1e-6)
            elif cfg.learning_rate_decrease == 'const':
                self._learning_rate = learning_rate_const
            else:
                self._learning_rate = tf.maximum(self.learning_rate, 1e-6)
            tf.summary.scalar('learning_rate', self._learning_rate)

            self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9)

    def inference(self, features, reuse, is_train):

        return
