import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np
import math


def spread_loss(labels, logits, thres_op, downweight=0.5):
    """ Implementation of spread loss based on marging loss,
    anealing the margin from 0.1 to 0.4 with step as configured

    :param labels:
    :param logits:
    :param thres_op:
    :param downweight:
    :return:
    """
    logits = logits-0.5

    positive_cost = labels*tf.cast(tf.less(logits, thres_op), tf.float32)*tf.square(logits-thres_op)
    negative_cost = (1-labels)*tf.cast(tf.greater(logits, -thres_op, tf.float32))*tf.square(logits+thres_op)
    return 0.5*positive_cost+downweight*0.5*negative_cost

def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
  """Penalizes deviations from margin for each logit.

  Each wrong logit costs its distance to margin. For negative logits margin is
  0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
  margin is 0.4 from each side.

  Args:
    labels: tensor, one hot encoding of ground truth.
    raw_logits: tensor, model predictions in range [0, 1]
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    downweight: scalar, the factor for negative cost.

  Returns:
    A tensor with cost for each data point of shape [batch_size].
  """
  logits = raw_logits - 0.5
  positive_cost = labels * tf.cast(tf.less(logits, margin), tf.float32) * tf.pow(logits - margin, 2)
  negative_cost = (1 - labels) * tf.cast(tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
  return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def top_1_error(outputs, labels, k=1):
    batch_size = tf.to_float(tf.shape(outputs)[0])
    top_k = tf.to_float(tf.nn.in_top_k(outputs, labels, k))
    num_correct = tf.reduce_sum(top_k)

    return (batch_size-num_correct)/batch_size


def top_2_error(outputs, labels):
    batch_size = tf.to_float(tf.shape(outputs)[0])
    max1_index = tf.one_hot(tf.argmax(outputs, axis=1), 10)
    outputs = outputs - tf.multiply(outputs, max1_index)
    max2_index = tf.one_hot(tf.argmax(outputs, axis=1), 10)
    max_index = max1_index + max2_index
    labels = tf.cast(labels, tf.float32)
    num = tf.reduce_sum(tf.multiply(max_index, labels), axis=1)
    num_correct = tf.reduce_sum(tf.cast(tf.equal(num, 2), tf.float32))
    return (batch_size - num_correct) / batch_size


# squash the inputs at the last axis
def squash(inputs, show=False):
    batch_size = int(inputs.get_shape()[0])

    if cfg.squash_norm == 'l2':
        norm_2 = tf.norm(inputs, axis=-1, keepdims=True)
        norm_squared = norm_2*norm_2
        outputs = (inputs/norm_2)*norm_squared/(1+norm_squared)

    return outputs


def squash_with_bias(inputs, is_train):
    num_caps = int(inputs.get_shape()[1])
    regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)
    biases = slim.variable('baises', shape=[num_caps, 16], regularizer=regularizer,
                           initializer=tf.constant_initializer(0.01), dtype=tf.float32, trainable=is_train)

    outputs = squash(inputs+biases)

    return outputs


def faster_matmul(a, b):
    shape_a = a.get_shape()

    reshape_a = []
    for iter in shape_a:
        reshape_a.append(int(iter))
    reshape_a.append(1)

    shape_b = b.get_shape()
    reshape_b = []
    for iter in shape_b:
        reshape_b.append(int(iter))
    reshape_b.insert(-2, 1)

    a = tf.reshape(a, shape=reshape_a)
    b = tf.reshape(b, shape=reshape_b)
    output = a*b

    output = tf.reduce_sum(output, axis=-2, keepdims=False)
    return output


# input should be a tensor with size as [batch_size, caps_num_in, channel_num_in]
def mat_transform(input, caps_num_out, is_train):
    batch_size = int(input.get_shape()[0])
    caps_num_in = int(input.get_shape()[1])

    std = math.sqrt(2./caps_num_out)

    regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)
    w = slim.variable('w', shape=[1, caps_num_out, caps_num_in, 4, 4], regularizer=regularizer,
                      initializer=tf.random_normal_initializer(mean=0.0, stddev=std), trainable=is_train)

    output = tf.reshape(input, [batch_size, 1, caps_num_in, 4, 4])
    output = tf.tile(output, [1, caps_num_out, 1, 1, 1])

    output = tf.reshape(faster_matmul(output, w), [batch_size, caps_num_out, caps_num_in, -1])

    return output


def relu_ge(inputs):
    if cfg.relu_type == 'leaky':
        outputs = tf.nn.leaky_relu(inputs, alpha=0.1, name='leaky_relu')
    elif cfg.relu_type == 'relu':
        outputs = tf.nn.relu(inputs)
    else:
        raise ValueError('Activation is error!!!')

    return outputs


def bn_relu_conv_layer(inputs, kernel_size, channel_out, stride, is_train, dropout_rate=-0.1, padding='SAME'):
    """
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param reuse:
    :param dropout_rate:
    :param channel_out:
    :param inputs: 4D tensor
    :param kernel_size:
    :param stride: stride size for conv
    :param is_train: training tag used in slim
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    """

    outputs = batch_norm_res(inputs, is_train)

    outputs = relu_ge(outputs)#tf.nn.relu(outputs)

    if dropout_rate > 0.0:
        outputs = slim.dropout(outputs, keep_prob=dropout_rate, is_training=is_train)

    outputs = conv2d_clean(outputs, channel_out, kernel_size, stride, is_train, padding=padding)
    return outputs


def conv2d_clean(inputs, num_output, kernel_size, stride, is_train, bias=False, padding='SAME'):
    regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)
    n_j = num_output*kernel_size*kernel_size
    std = math.sqrt(2./n_j)

    if bias:
        outputs = slim.conv2d(inputs, num_outputs=num_output, kernel_size=[kernel_size, kernel_size], stride=stride,
                              trainable=is_train, activation_fn=None,
                              weights_initializer=tf.random_normal_initializer(mean=0, stddev=std),
                              weights_regularizer=regularizer, biases_regularizer=regularizer, padding=padding)
    else:
        outputs = slim.conv2d(inputs, num_outputs=num_output, kernel_size=[kernel_size, kernel_size], stride=stride,
                              trainable=is_train, weights_initializer=tf.random_normal_initializer(mean=0, stddev=std),
                              biases_initializer=None, activation_fn=None,
                              weights_regularizer=regularizer, padding=padding)

    return outputs


def batch_norm_res(inputs, is_train):
    outputs = slim.batch_norm(inputs, is_training=is_train, scale=True, trainable=is_train, epsilon=1e-05)

    return outputs


def wide_residual_block(inputs, num_output_channel, stride, is_train):
    num_input_channel = inputs.get_shape().as_list()[-1]

    if not num_input_channel == num_output_channel:
        with tf.variable_scope('conv1'):
            inputs = relu_ge(batch_norm_res(inputs, is_train))
            outputs = conv2d_clean(inputs, num_output_channel, 3, stride, is_train)

        with tf.variable_scope('conv1_1'):
            inputs = conv2d_clean(inputs, num_output_channel, 1, stride, is_train, padding='VALID')
    else:
        with tf.variable_scope('conv1'):
            outputs = bn_relu_conv_layer(inputs, 3, num_output_channel, 1, is_train)

    with tf.variable_scope('brc1'):
        outputs = bn_relu_conv_layer(outputs, 3, num_output_channel, 1, is_train, cfg.dropout_rate)

    outputs = outputs+inputs
    return outputs


def routing_as_recon(pose, is_train):
    batch_size = int(pose.get_shape()[0])
    caps_num_in = int(pose.get_shape()[2])
    caps_num_out = int(pose.get_shape()[1])

    # calculate y_j given all r_ji = 1 as the first step of iteration 0
    y = tf.reduce_sum(pose, axis=2, keepdims=True)

    for i in range(cfg.iter_routing-1):
        y = y/tf.norm(y, axis=-1, keepdims=True)

        r = tf.reduce_sum(pose*y, axis=-1, keepdims=True)/2

        y = tf.reduce_sum(pose*r, axis=2, keepdims=True)/tf.reduce_sum(tf.square(r), axis=2, keepdims=True)


    k = slim.variable('sc_k', shape=[caps_num_out, 1, 16], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), trainable=is_train)
    b = slim.variable('sc_b', shape=[caps_num_out, 1, 1], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01), trainable=is_train)
    activation_out = tf.reduce_sum(k*y, axis=-1, keepdims=True)+b

    output = tf.squeeze(y)
    activation_out = tf.squeeze(activation_out)
    return output, activation_out


def routing_as_recon_clean(pose, is_train):
    batch_size = int(pose.get_shape()[0])
    caps_num_in = int(pose.get_shape()[2])
    caps_num_out = int(pose.get_shape()[1])

    y = tf.reduce_sum(pose, axis=2, keepdims=True)

    for i in range(cfg.iter_routing-1):
        y = y/tf.norm(y, axis=-1, keepdims=True)

        r = tf.reduce_sum(pose*y, axis=-1, keepdims=True)/2

        y = tf.reduce_sum(pose*r, axis=2, keepdims=True)/tf.reduce_sum(tf.square(r), axis=2, keepdims=True)

    outputs = tf.squeeze(y)#
    return outputs


def em_routing(pose, activation, is_train):
    batch_size = int(pose.get_shape()[0])
    caps_num_in = int(pose.get_shape()[2])
    caps_num_out = int(pose.get_shape()[1])

    r = tf.constant(np.ones([batch_size, caps_num_out, caps_num_in], dtype=np.float32)/caps_num_out)
    activation = tf.reshape(activation, [batch_size, 1, -1])

    beta_v = slim.variable('beta_v', shape=[caps_num_out, 1, 16], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01), trainable=is_train)
    beta_a = slim.variable('beta_a', shape=[caps_num_out, 1, 1], dtype=tf.float32,
                           initializer=tf.constant_initializer(0.01), trainable=is_train)

    for i in range(cfg.iter_routing):
        # m-step
        r1 = tf.reshape(r*activation, [batch_size, caps_num_out, caps_num_in, 1])
        r1_sum = tf.reduce_sum(r1, axis=2, keepdims=True)
        r1 = r1/(r1_sum+cfg.epsilon)

        miu = tf.reduce_sum(r1 * pose, axis=2, keepdims=True) #batch_size, caps_num_out, caps_num_in=1, 16
        sigma_square = tf.reduce_sum(r1*tf.square(pose-miu), axis=2, keepdims=True) #batch_size, caps_num_out, caps_num_in=1, 16

        activation_out = tf.reduce_sum(beta_v+tf.log(tf.sqrt(sigma_square)), axis=-1, keepdims=True)*r1_sum #batch_size, caps_num_out, caps_num_in=1, 1
        activation_out = tf.nn.sigmoid(np.power(10.0, i-3.0)*(beta_a-activation_out)) #batch_size, caps_num_out, caps_num_in=1, 1

        # e_step
        if i < cfg.iter_routing-1:
            log_p = -tf.log(tf.sqrt(sigma_square))-tf.square(pose-miu)/(2*sigma_square)
            log_p = log_p-(tf.reduce_max(log_p, axis=[1, 3], keepdims=True)-tf.log(10.0))
            p = tf.exp(tf.reduce_sum(log_p, -1, keepdims=True))

            r = activation_out*p
            r = tf.squeeze(r/(tf.reduce_sum(r, axis=1, keepdims=True)+cfg.epsilon))

    pose = tf.squeeze(miu) #batch_size, caps_num_out, 16
    activation = tf.squeeze(activation_out) #batch_size, caps_num_out
    return pose, activation


# input should be a tensor with size as [batch_size, caps_num_out, caps_num_in, channel_num]
def dynamic_routing(inputs, is_train):
    batch_size = int(inputs.get_shape()[0])
    caps_num_in = int(inputs.get_shape()[2])
    caps_num_out = int(inputs.get_shape()[1])
    channel_num = int(inputs.get_shape()[-1])

    inputs_transposed = tf.transpose(inputs, [0, 1, 3, 2])
    input_stopped = tf.stop_gradient(inputs, name='stop_gradient')

    b = tf.fill([batch_size, caps_num_out, caps_num_in, 1], 0.0)

    biases = slim.variable('biases', shape=[caps_num_out, channel_num, 1], initializer=tf.constant_initializer(0.1),
                           dtype=tf.float32, trainable=is_train)
    test_vals = tf.global_variables()

    for i in range(cfg.iter_routing):
        if cfg.leaky:
            c = leaky_softmax(b, caps_num_out, dim=1)
        else:
            c = tf.nn.softmax(b, dim=1)

        if i == cfg.iter_routing-1:
            s = faster_matmul(inputs_transposed, c)
            v = squash(tf.squeeze(s+biases))
        else:
            s = faster_matmul(inputs_transposed, c)
            v = squash(tf.squeeze(s+biases))
            b += tf.reduce_sum(tf.reshape(v, shape=[batch_size, caps_num_out, 1, -1])*inputs, axis=-1, keepdims=True)

    return v


# input should be a tensor with size as [batch_size, width, height, num_capsules, dim]
def primary_routing(inputs, is_train):
    batch_size = int(inputs.get_shape()[0])
    data_size = int(inputs.get_shape()[1])
    num_capsules = int(inputs.get_shape()[3])
    dim = int(inputs.get_shape()[-1])

    b = tf.fill([batch_size, data_size, data_size, num_capsules, 1], 0.0)
    biases = slim.variable('baises', shape=[cfg.num_capsule_primary, dim], initializer=tf.constant_initializer(0.1),
                           dtype=tf.float32, trainable=is_train)
    test_vals = tf.global_variables()

    for i in range(cfg.primary_routing_num):
        if cfg.leaky:
            c = leaky_softmax(b, num_capsules, dim=3)
        else:
            c = tf.nn.softmax(b, dim=3)

        if i == cfg.primary_routing_num - 1:
            s = inputs * c
            v = squash(s + biases)
        else:
            s = inputs*c
            v = squash(s + biases)
            b += tf.reduce_sum(v * inputs, axis=-1, keepdims=True)

    return v


def leaky_softmax(inputs, num_output, dim):
    leak = tf.zeros_like(inputs)
    leak = tf.reduce_sum(leak, axis=dim, keepdims=True)
    leaky_inputs = tf.concat([inputs, leak], axis=dim)
    output = tf.nn.softmax(leaky_inputs, dim=dim)

    return tf.split(output, [num_output, 1], axis=dim)[0]


def fc_res(inputs, num_classes, is_train):
    regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.weight_decay)
    input_channel = int(inputs.get_shape()[-1])
    std = 1./math.sqrt(input_channel)

    outputs = slim.fully_connected(inputs, num_classes, activation_fn=None,
                         weights_initializer=tf.random_uniform_initializer(minval=-std, maxval=std),
                         weights_regularizer=regularizer, biases_initializer=tf.zeros_initializer(),
                         biases_regularizer=regularizer, trainable=is_train)

    return outputs