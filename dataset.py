import tensorflow as tf
import os
import numpy as np

from config import cfg

def get_dataset_size_train(dataset_name):
    options = {'mnist': 55000, 'smallNORB': 23400, 'cifar10': 50000, 'cifar10_original': 50000, 'SVHN': 604388}
    return options[dataset_name]


def get_dataset_size_val(dataset_name: str):
    options = {'mnist': 10000, 'smallNORB': 23400, 'cifar10': 10000, 'cifar10_original': 10000, 'SVHN': 26032}
    return options[dataset_name]


def get_num_classes(dataset_name: str):
    options = {'mnist': 10, 'smallNORB': 5, 'cifar10': 10, 'SVHN': 10}
    return options[dataset_name]


def _read_input(filename_queue):
  """Reads a single record and converts it to a tensor.

  Each record consists the 3x32x32 image with one byte for the label.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
      image: a [32, 32, 3] float32 Tensor with the image data.
      label: an int32 Tensor with the label in the range 0..9.
  """
  label_bytes = 1
  height = 32
  depth = 3
  image_bytes = height * height * depth
  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, byte_data = reader.read(filename_queue)
  uint_data = tf.decode_raw(byte_data, tf.uint8)

  label = tf.cast(tf.strided_slice(uint_data, [0], [label_bytes]), tf.int32)
  label.set_shape([1])

  depth_major = tf.reshape(
      tf.strided_slice(uint_data, [label_bytes], [record_bytes]),
      [depth, height, height])
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def load_cifar10(is_train):
    if is_train:
        filenames = [os.path.join(cfg.cufar10_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    else:
        filenames = [os.path.join(cfg.cufar10_dir, 'test_batch.bin')]

    filename_queue = tf.train.string_input_producer(filenames)
    float_image, label = _read_input(filename_queue)

    return float_image, label


def load_mnist(is_training):
    if is_training:
        split = 'train'
        shift = 2
    else:
        split = 'test'
        shift = 0
    file_format = '{}_{}shifted_mnist.tfrecords'
    filenames = [os.path.join(cfg.mnist_dir, file_format.format(split, shift))]

    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=is_training)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    return features

def load_smallNORB(filenames, epochs: int):
    assert isinstance(filenames, list)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [2, 96, 96])
    img = tf.transpose(img, [1, 2, 0])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label


def create_inputs(dataset_name, is_train, epochs=20):
    features = []
    if is_train:
        for i in range(cfg.num_gpus):
            with tf.device('/cpu:0'):
                if dataset_name == 'cifar10':
                    features.append(create_inputs_cifar10(is_train=is_train))
                elif dataset_name == 'mnist':
                    features.append(create_inputs_mnist(is_train=is_train))
                elif dataset_name == 'smallNORB':
                    features.append(create_inputs_smallNORB(is_train=is_train))
                elif dataset_name == 'SVHN':
                    features.append(create_inputs_SVHN(is_train=is_train))
                else:
                    raise ValueError(
                        'Unexpected dataset {!r}, must be mnist, norb, cifar10.'.format(
                            dataset_name))
    else:
        with tf.device('/cpu:0'):
            if dataset_name == 'cifar10':
                features.append(create_inputs_cifar10(is_train=is_train))
            elif dataset_name == 'mnist':
                features.append(create_inputs_mnist(is_train=is_train))
            elif dataset_name == 'smallNORB':
                features.append(create_inputs_smallNORB(is_train=is_train))
            elif dataset_name == 'SVHN':
                features.append(create_inputs_SVHN(is_train=is_train))
            else:
                raise ValueError(
                    'Unexpected dataset {!r}, must be mnist, norb, cifar10.'.format(
                        dataset_name))
    return features


def create_inputs_mnist(is_train, image_size=28):
    channels = 1
    num_class = 10
    features = load_mnist(is_train)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 1])
    image.set_shape([image_size, image_size, 1])

    image = tf.cast(image, tf.float32)*(1./255.)
    label = tf.cast(features['label'], tf.int32)
    dataset_name = 'mnist'
    return get_batch_features(image, label, image, cfg.batch_size, is_train, image_size, num_class, channels, dataset_name)


def load_SVHN(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    test, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def create_inputs_SVHN(is_train):
    dataset_name = 'SVHN'
    image_size = 32
    channels = 3
    num_class = 10
    if is_train:
        filenames = ['./data/SVHN/SVHN_extra_and_train.tfrecords']
    else:
        filenames = ['./data/SVHN/SVHN_test.tfrecords']
    image, label = load_SVHN(filenames)
    recons_image = image/255.0
    x_std = (recons_image - np.array([109.9, 109.7, 113.8], dtype=np.float32) / 255.0) / (
        np.array([50.1, 50.6, 50.8], dtype=np.float32) / 255.0)

    return get_batch_features(x_std, label, recons_image, cfg.batch_size, is_train, image_size, num_class, channels, dataset_name)


def create_inputs_cifar10(is_train):
    image_size = 32
    channels = 3
    num_class = 10

    x, y = load_cifar10(is_train)

    if is_train:
        x = tf.image.resize_image_with_crop_or_pad(x, image_size+8, image_size+8)
        x = tf.random_crop(x, [image_size, image_size, channels])
        x = tf.image.random_flip_left_right(x)
    recons_image = x / 255.0
    x_std = (recons_image - np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255.0) / (
        np.array([63.0, 62.1, 66.7], dtype=np.float32) / 255.0)
    dataset_name = 'cifar10'

    return get_batch_features(x_std, y, recons_image, cfg.batch_size, is_train, image_size, num_class, channels, dataset_name)


def create_inputs_smallNORB(is_train):

    if is_train:
        filenames = ['data/smallnorb/train0.tfrecords']
    else:
        filenames = ['data/smallnorb/test0.tfrecords']

    image, label = load_smallNORB(filenames, 600)

    image = tf.image.resize_images(image, [48, 48])
    params_shape = [image.get_shape()[-1]]
    if is_train:
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.random_crop(image, [32, 32, 2])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    image = tf.image.per_image_standardization(image)
    image_size = 32
    num_class = 5
    channels = 2
    recons_image = image
    dataset_name = 'smallNORB'
    return get_batch_features(image, label, recons_image, cfg.batch_size, is_train, image_size, num_class, channels,
                              dataset_name)


def get_batch_features(image, label, recons_image, batch_size, is_train, image_size, num_class, channels, dataset_name):
    # image = tf.transpose(image, [2, 0, 1])
    if dataset_name == 'multimnist':
        features = {
            'images': image,
            'labels': label,
            'recons_image': image,
            'recons_label': label,
        }
    else:
        features = {
            'images': image,
            'labels': tf.one_hot(label, num_class, dtype=tf.float32),
            'recons_image': recons_image,
            'recons_label': label,
        }

    if is_train:
        batched_features = tf.train.shuffle_batch(
            features,
            batch_size=batch_size,
            num_threads=16,
            capacity=10000 + 3 * batch_size,
            min_after_dequeue=10000)
    else:
        batched_features = tf.train.batch(
            features,
            batch_size=batch_size,
            num_threads=1,
            capacity=10000 + 3 * batch_size)

    batched_features['labels'] = tf.reshape(batched_features['labels'], [batch_size, num_class])
    batched_features['height'] = image_size
    batched_features['depth'] = channels
    batched_features['num_classes'] = num_class
    if dataset_name == 'multimnist':
        batched_features['recons_label'] = tf.reshape(batched_features['recons_label'], [batch_size, num_class])
        batched_features['num_targets'] = 2
    else:
        batched_features['recons_label'] = tf.reshape(batched_features['recons_label'], [batch_size])
        batched_features['num_targets'] = 1

    return batched_features



