# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Basic Model class which provides the optimization wrap around inference.

A general model class wrapps the loss and optimizer operations around the
inference graph. Therefore, it should define the learning rate, global step
and optimizer. The current version supports a multiple gpu scenario by
enforcing single gpu to select one gpu for calculations and reuse variables.

Different models will only have different inference graphs and they share the
training and evaluation ops. Therefore, we define the inference function
as an abstract method that each model should define specifically for itself.
"""
import abc
import collections

import tensorflow as tf
import tensorflow.contrib.slim as slim

import nn_utils as nnu
from config import cfg

TowerResult = collections.namedtuple('TowerResult', ('inferred', 'grads'))
JoinedResult = collections.namedtuple('JoinedResult', ('summary', 'train_op'))
Inferred = collections.namedtuple('Inferred', ('logits', 'remakes'))


class Model(object):
    """Base class for building a model and running inference on it."""

    __metaclass__ = abc.ABCMeta

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

            self._optimizer = tf.train.AdamOptimizer(self._learning_rate)

    def evaluate(self, inferred, features, scope):
        logits = inferred.logits
        labels = features['labels']
        num_targets = features['num_targets']

        with tf.name_scope('loss'):
            if cfg.loss_type == 'sigmoid':
                classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels / 2.0, logits=logits)
            elif cfg.loss_type == 'softmax':
                classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=features['recons_label'], logits=logits)
            elif cfg.loss_type == 'margin':
                classification_loss = nnu.margin_loss(labels=labels, raw_logits=logits)
            elif cfg.loss_type == 'spread':
                classification_loss = nnu.spread_loss(labels=labels, logits=logits, thres_op=self.thres_op)
            else:
                raise NotImplementedError('Not implemented')

            with tf.name_scope('total'):
                batch_classification_loss = tf.reduce_mean(classification_loss)
                tf.add_to_collection('losses', batch_classification_loss)
        tf.summary.scalar('classification_loss', batch_classification_loss)

        reg_loss = []
        if cfg.weight_reg:
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        all_losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(all_losses + reg_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                _, targets = tf.nn.top_k(labels, k=num_targets)
                if cfg.loss_type == 'softmax':
                    logits = tf.nn.softmax(logits)
                _, predictions = tf.nn.top_k(logits, k=num_targets)
                missed_targets = tf.contrib.metrics.set_difference(targets, predictions)
                num_missed_targets = tf.contrib.metrics.set_size(missed_targets)
                correct = tf.equal(num_missed_targets, 0)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        return total_loss

    @abc.abstractmethod
    def inference(self, features, reuse, is_train):
        """Adds the inference graph ops.

        Builds the architecture of the neural net to derive logits from features.
        The inference graph defined here should involve trainable variables
        otherwise the optimizer will raise a ValueError.

        Args:
          features: Dictionary of batched feature tensors like images and labels.
          reuse:
          is_train:
        Returns:
          An Inferred named tuple for expected outputs of the model like 'logits'
          and 'remakes' for the reconstructions.
        """
        raise NotImplementedError('Not implemented')

    def _single_tower(self, tower_ind, feature, reuse=False):
        """Calculates the model gradient for one tower.

        Adds the inference and loss operations to the graph. Calculates the
        gradients based on the loss. Appends all the output values of this tower to
        their respective lists.

        Args:
          tower_ind: The index number for this tower. Each tower is named as
                      tower_{tower_ind} and resides on gpu:{tower_ind}.
          feature: Dictionary of batched features like images and labels.
        Returns:
          A namedtuple TowerResult containing the inferred values like logits and
          reconstructions, gradients and evaluation metrics.
        """
        with tf.device('/gpu:%d' % tower_ind):
            with tf.name_scope('tower_%d' % (tower_ind)) as scope:
                with slim.arg_scope([slim.variable], device='/cpu:0'):
                    inferred = self.inference(feature, reuse=reuse, is_train=True)

                losses = self.evaluate(inferred=inferred, features=feature, scope=scope)
                # tf.get_variable_scope().reuse_variables()
                grads = self._optimizer.compute_gradients(losses)
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return TowerResult(inferred, grads)

    def _average_updates(self, tower_update_ops):
        average_updates = []
        for update_ops in zip(*tower_update_ops):
            update_op = tf.stack([u for u in update_ops])
            update_op = tf.reduce_mean(update_op, 0)

            average_updates.append(update_op)
        return average_updates

    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each variable across all towers.

        Args:
          tower_grads: List of gradient lists for each tower. Each gradient list
            is a list of (gradient, variable) tuples for all variables.
        Returns:
          List of pairs of (gradient, variable) where the gradient has been
          averaged across all towers.
        """
        average_grads = []
        for grads_and_vars in zip(*tower_grads):
            grads = tf.stack([g for g, _ in grads_and_vars])
            grad = tf.reduce_mean(grads, 0)

            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _summarize_towers(self, tower_grads, update_ops):
        """Aggregates the results and gradients over all towers.

        Args:
          almosts: The number of almost correct samples for each tower.
          corrects: The number of correct samples for each tower.
          tower_grads: The gradient list for each tower.

        Returns:
          A JoinedResult of evaluation results, the train op and the summary op.
        """

        grads = self._average_gradients(tower_grads)

        with tf.control_dependencies(update_ops):
            train_op = self._optimizer.apply_gradients(grads, global_step=self._global_step)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary = tf.summary.merge(summaries)
        return JoinedResult(summary, train_op)

    def multi_gpu(self, features, num_gpus):
        """Build the Graph and add the train ops on multiple GPUs.

        Divides the inference and gradient computation on multiple gpus.
        Then aggregates the gradients and return the resultant ops.

        Args:
          features: A list of dictionary of different features of input data.
                    len(features) should be at least num_gpus.
          num_gpus: Number of gpus to be distributed on.
        Returns:
          A tuple of JoinedResult output Ops to be called in Session.run for
          training, evaluation or visualization, such as train_op and merged
          summary and a list of inferred outputs of each tower.
        """
        inferred = []
        tower_grads = []
        reuse = False
        for i in range(num_gpus):
            tower_output = self._single_tower(i, features[i], reuse)
            inferred.append(tower_output.inferred)
            tower_grads.append(tower_output.grads)
            reuse = True

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        summarized_results = self._summarize_towers(tower_grads, update_ops)
        return summarized_results, inferred

    def validation(self, features):
        features = features[0]
        with tf.device('gpu:0'):
            with tf.name_scope('tower_val') as scope:
                with slim.arg_scope([slim.variable], device='/cpu:0'):
                    inferred = self.inference(features, reuse=True, is_train=False)

                logits = inferred.logits
                if cfg.loss_type == 'softmax':
                    logits = tf.nn.softmax(logits)
                if features['num_targets'] == 1:
                    error_val = nnu.top_1_error(logits, features['recons_label'])
                else:
                    error_val = nnu.top_2_error(logits, features['recons_label'])

        return error_val

    def remake_evaluate(self, capsule_output, features, reuse, is_train):
        num_pixel = features['depth'] * features['height'] * features['height']

        mask = tf.expand_dims(features['labels'], -1)
        mask = tf.tile(mask, [1, 1, 16])

        inputs = capsule_output * mask
        inputs = tf.contrib.layers.flatten(inputs)

        with tf.variable_scope('recons', reuse=reuse) as scope:
            if cfg.remake_type == 'fc':
                reconstruction = nnu.remake_fc(inputs, num_pixel, reuse=reuse, is_train=is_train)
            elif cfg.remake_type == 'conv':
                reconstruction = nnu.remake_conv(inputs, num_pixel, reuse=reuse, is_train=is_train)
            else:
                reconstruction = None

            with tf.name_scope('loss') as scope:
                image = tf.contrib.layers.flatten(features['recons_image'])
                distance = tf.pow(reconstruction-image, 2)
                loss = tf.reduce_sum(distance, axis=-1)
                batch_loss = tf.reduce_mean(loss)
                balanced_loss = 0.0005*batch_loss
                tf.add_to_collection('losses', balanced_loss)
                tf.summary.scalar('reconstruction_error', balanced_loss)

        # TODO: show images in tensorboard
        return reconstruction
