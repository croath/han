import argparse
import sys
import tempfile
import log_helper
from log_helper import log
from data_reader import read_data_sets
from data_reader import get_real_images
from data_reader import dense_to_one_hot
import data_reader
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime
import numpy as np
import uuid
import os
from chn_converter import int_to_chinese

np.set_printoptions(threshold=np.inf)

log_helper.initLogging('log/' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '.log')

FLAGS = None

def deepnn(top_k):
    batch_norm_params = {'is_training': True, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='images')
        labels = tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.charater_num], name='labels')
        conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
        max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
        conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
        max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
        conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
        max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')
        conv_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv4')
        conv_5 = slim.conv2d(conv_4, 1024, [3, 3], padding='SAME', scope='conv5')
        conv_6 = slim.conv2d(conv_5, 1024, [3, 3], padding='SAME', scope='conv6')
        conv_7 = slim.conv2d(conv_6, 1024, [3, 3], padding='SAME', scope='conv7')
        max_pool_7 = slim.max_pool2d(conv_7, [2, 2], [2, 2], padding='SAME')

        flatten = slim.flatten(max_pool_7)
        fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 2048, activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charater_num, activation_fn=None, scope='fc2')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int64), tf.argmax(labels, 1)), tf.float32))

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        rate = tf.train.exponential_decay(1e-4, global_step, decay_steps=2000, decay_rate=0.99, staircase=True)
        train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)
        probabilities = tf.nn.softmax(logits)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        # print(tf.nn.in_top_k(probabilities, y_, top_k).shape)
        # accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.cast(tf.argmax(probabilities, 1), tf.float32), tf.argmax(y_, 1), top_k), tf.float32))

    return {'images': images,
            'labels':labels,
            'logits': logits,
            'keep_prob': keep_prob,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'probabilities': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k,
            'merged_summary_op': merged_summary_op,
            'global_step': global_step}

def main(_):
  valid_data = read_data_sets(FLAGS.valid_dir, FLAGS.labellist)

  d = deepnn(1)

  graph_location = tempfile.mkdtemp() if FLAGS.graph_dir == None else os.path.join(FLAGS.graph_dir, str(uuid.uuid4()))
  log('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  session_config = tf.ConfigProto()
  session_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

  with tf.Session(config=session_config) as sess:
      saver = tf.train.Saver()

      if FLAGS.read_from_checkpoint:
          ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
          if ckpt:
              saver.restore(sess, ckpt)
          else:
              sess.run(tf.global_variables_initializer())
      else:
          sess.run(tf.global_variables_initializer())

      if FLAGS.mode == "train":
          train_data = read_data_sets(FLAGS.data_dir, FLAGS.labellist)

          for i in range(FLAGS.epoch_num):
              inside_step = 0
              while not train_data.epochs_completed:
                  batch = train_data.next_batch(FLAGS.batch_size)
                #   if i % (FLAGS.batch_size // 10) == 0:
                    #   train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                  _, loss_val, summary, step, acc = sess.run([d['train_op'], d['loss'], d['merged_summary_op'], d['global_step'], d['accuracy']], feed_dict={d['images']: batch[0].reshape([-1, 64, 64, 1]), d['labels']: batch[1], d['keep_prob']: 0.5})
                  train_writer.add_summary(summary, step)
                  if inside_step % 50 == 0:
                      log('step %d, training accuracy %g loss is %g'% (inside_step, acc, loss_val))
                  inside_step += 1

              train_data.restart_epoch()
              saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=i+1)

              valid_acc = 0.0
              valid_step = 0
              while not valid_data.epochs_completed:
                  batch = valid_data.next_batch(FLAGS.batch_size)
                  _, acc_val = sess.run([ d['logits'], d['accuracy']], feed_dict={d['images']: batch[0].reshape([-1, 64, 64, 1]), d['labels']: batch[1], d['keep_prob']: 1.0})
                  valid_acc += acc_val
                  valid_step += 1

              valid_data.restart_epoch()
              valid_acc = valid_acc / valid_step
              log('epoch %d valid accuracy %g' % (i, acc_val))
      elif FLAGS.mode == "test":
          valid_acc = 0.0
          valid_step = 0
          while not valid_data.epochs_completed:
              batch = valid_data.next_batch(FLAGS.batch_size)
              labels_val, logits_val, acc_val = sess.run([d['labels'], d['logits'], d['accuracy']], feed_dict={d['images']: batch[0].reshape([-1, 64, 64, 1]), d['labels']: batch[1], d['keep_prob']: 1.0})
              for i in range(0, len(labels_val)):
                  label = labels_val[i]
                  logit = logits_val[i]

                  input_char = int_to_chinese(data_reader.unique_label_list[label.argmax()])
                  output_char = int_to_chinese(data_reader.unique_label_list[logit.argmax()])

                  match = True if input_char == output_char else False
                  log('Input: %s\tOuput: %s\t%r' %(input_char, output_char, match))

              valid_acc += acc_val
              valid_step += 1

          valid_acc = valid_acc / valid_step
          log('Valid accuracy %g' % acc_val)

          while True:
              log("Input the testing path of images or the parent directory:\n")
              sentence = sys.stdin.readline(1)
              test_data = read_data_sets(sentence, FLAGS.labellist)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/Users/croath/Desktop/sample/data2', help='Directory for storing input data')
  parser.add_argument('--valid_dir', type=str, default='/Users/croath/Desktop/sample/data3', help='Directory for storing input data')
  parser.add_argument('--checkpoint_dir', type=str, default='/Users/croath/Desktop/checkpoint/', help='Directory for stroing checkpoint')
  parser.add_argument('--graph_dir', type=str, help='Directory to save graph')
  parser.add_argument('--read_from_checkpoint', type=bool, default=False, help='Load from a checkpoint or not')
  parser.add_argument('--charater_num', type=int, default=205, help='How many unique characters you have')
  parser.add_argument('--mode', type=str, default='train', help='Running mode')
  parser.add_argument('--labellist', type=str, default=None, help='Labels list')
  parser.add_argument('--epoch_num', type=int, default=10, help='Labels list')
  parser.add_argument('--batch_size', type=int, default=200, help='Labels list')
  parser.add_argument('--gpu_fraction', type=float, default=1.0, help='Percent of GPU usage')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
