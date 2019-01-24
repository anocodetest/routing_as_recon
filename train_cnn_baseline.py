import tensorflow as tf
import sys
import os
import time
import numpy as np

from config import cfg
import dataset
from model_cnn_baseline import Cnn_Net


def main(_):
    dataset_name = cfg.dataset
    dataset_size_train = dataset.get_dataset_size_train(dataset_name)
    dataset_size_val = dataset.get_dataset_size_val(dataset_name)

    num_batch_per_epoch_train = dataset_size_train // (cfg.batch_size*cfg.num_gpus) \
        if dataset_size_train % (cfg.batch_size*cfg.num_gpus) == 0 \
        else dataset_size_train // (cfg.batch_size*cfg.num_gpus) + 1
    num_batch_per_epoch_val = dataset_size_val // (cfg.batch_size*cfg.num_gpus) \
        if dataset_size_val % (cfg.batch_size*cfg.num_gpus) == 0 \
        else dataset_size_val // (cfg.batch_size*cfg.num_gpus) + 1

    with tf.Graph().as_default():
        inputs_train = dataset.create_inputs(dataset_name, is_train=True)
        inputs_val = dataset.create_inputs(dataset_name, is_train=False)

        model = Cnn_Net()
        # build train network on multiple gpus
        train_ops, _ = model.multi_gpu(inputs_train, cfg.num_gpus)
        # build validation network on gpu 0
        val_ops = model.validation(inputs_val)

        test_ops = tf.get_collection('my_test')
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                                                            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        if not os.path.exists(os.path.join('logdir', cfg.date)):
            os.makedirs(os.path.join('logdir', cfg.date))
        summary_writer = tf.summary.FileWriter(os.path.join(
            'logdir', cfg.date, cfg.logdir))

        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        session.run(init_op)
        saver = tf.train.Saver(max_to_keep=30)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        learning_rate = cfg.init_learning_rate
        times = []
        for step in range(cfg.epoch * num_batch_per_epoch_train):
            tic = time.time()

            if step > 0 and step % num_batch_per_epoch_train == 0:
                error_list = []
                times = []

                for step_val in range(num_batch_per_epoch_val):
                    error_val_value = session.run(val_ops)
                    error_list.append(error_val_value)

                error_val_value = np.mean(error_list)
                summ_val = tf.Summary()
                summ_val.value.add(tag='full_val_error', simple_value=error_val_value.astype(np.float))
                summary_writer.add_summary(summ_val, step)
                summary_writer.flush()

            try:
                if test_ops:
                    test = session.run(test_ops)
                    print(test)

                session.run(train_ops.train_op, feed_dict={model.learning_rate: learning_rate})
                times.append(time.time() - tic)
                print('%d iteration finished in ' % step + '%f second,' % np.mean(times))
            except KeyboardInterrupt:
                session.close()
                sys.exit()
            except tf.errors.InvalidArgumentError:
                print('iteration contains NaN gradients. Discard.')
                continue

            if step > 0 and step % 50 == 0:
                summary_str = session.run(train_ops.summary, feed_dict={model.learning_rate: learning_rate})
                summary_writer.add_summary(summary_str, step)

            if step > 0 and step % num_batch_per_epoch_train == 0:
                ckpt_path = os.path.join(
                'logdir', cfg.date, cfg.logdir, 'model.ckpt')
                saver.save(session, ckpt_path, global_step=step)

                if step >= cfg.epoch*num_batch_per_epoch_train*cfg.decay_prop_0:
                    learning_rate = cfg.init_learning_rate*cfg.decay_factor

                if step >= cfg.epoch*num_batch_per_epoch_train*cfg.decay_prop_1:
                    learning_rate = cfg.init_learning_rate*cfg.decay_factor*cfg.decay_factor

        coord.request_stop()
        coord.join(threads)
        session.close()
        summary_writer.close()


if __name__ == "__main__":
    tf.app.run()