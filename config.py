import tensorflow as tf

flags = tf.app.flags

############################
#   environment setting    #
############################
flags.DEFINE_boolean('is_train', True, 'train or predict phase')
flags.DEFINE_string('logdir', '0', 'logs directory')
flags.DEFINE_string('test_logdir', '0_test', 'test logs directory')
flags.DEFINE_string('date', '20180701', 'the date of experiment')

flags.DEFINE_string('dataset', 'mnist', 'the name of dataset')
flags.DEFINE_string('cufar10_dir', './data/cifar-10-batches-bin', 'directory to cifar10')
flags.DEFINE_string('mnist_dir', './data/mnist', 'directory to mnist')

############################
# common hyper parameters #
############################
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
flags.DEFINE_integer('batch_size', 100, 'batch size for one gpu')
flags.DEFINE_integer('epoch', 200, 'epoch')
flags.DEFINE_float('init_learning_rate', 0.001, '''Initial learning rate''')
flags.DEFINE_string('learning_rate_decrease', 'exp', 'the strategy to decrease the learning rate, exp, const, stage')
flags.DEFINE_integer('decay_steps', 2000, 'step to decay learning rate')
flags.DEFINE_float('exp_decay_rate', 0.96, 'decay rate in exponential decay')
flags.DEFINE_float('decay_factor', 0.1, '''How much to decay the learning rate each time''')
flags.DEFINE_float('decay_prop_0', 0.5, '''At which step to decay the learning rate''')
flags.DEFINE_float('decay_prop_1', 0.75, '''At which step to decay the learning rate''')
flags.DEFINE_float('decay_prop_2', -0.1, '''At which step to decay the learning rate''')
flags.DEFINE_string('loss_type', 'margin', 'sigmoid, softmax, margin, spread')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_boolean('weight_reg', False, 'train with regularization of weights')
flags.DEFINE_float('weight_decay', 0.0005, '''scale for l2 regularization''')
flags.DEFINE_integer('conv1_channel_num', 256, 'channel number of conv1 layer')
flags.DEFINE_string('relu_type', 'leaky', 'type of relu function, leaky or relu')

############################
# resnet parameters #
############################
flags.DEFINE_integer('num_res_block', 9, 'number of residual blocks at each scale')

############################
# wide resnet parameters # (default is for SVHN)
############################
flags.DEFINE_float('dropout_rate', 0.4, '''The rate for dropout''')
flags.DEFINE_integer('widen_factor', 8, 'k for widen the output channels')

############################
# capsule parameters, here parameters are adaptive to minst and smallnorb #
############################
flags.DEFINE_integer('iter_routing', 3, 'number of iterations')
flags.DEFINE_boolean('remake', False, 'whether use reconstruction')
flags.DEFINE_string('remake_type', 'fc', 'network structure for reconstruction')
flags.DEFINE_integer('num_capsule_primary', 32, 'number of capsules on primary')
flags.DEFINE_string('padding', 'VALID', 'padding for conv layers')
flags.DEFINE_boolean('primary_routing', True, 'whether routing at the primiary capsule layer')
flags.DEFINE_integer('primary_routing_num', 1, 'number of iterations in primary routing')
flags.DEFINE_string('squash_norm', 'l2', 'the norm type of squash')
flags.DEFINE_boolean('leaky', False, 'whether using leaky softmax or standard softmax')
flags.DEFINE_float('thres_schedule', 0.2, 'threshold will get to thres_max at current epoch')
flags.DEFINE_integer('routing_widen_factor', '2', 'number of channels after routing')

############################
# em_routing parameters #
############################
flags.DEFINE_float('ac_lambda0', 0.001, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 1.0, 'It is described that \lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.')



cfg = tf.app.flags.FLAGS
