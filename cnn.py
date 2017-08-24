from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import log_helper
from log_helper import log
from data_reader import read_data_sets
from data_reader import get_real_images
from data_reader import dense_to_one_hot
import tensorflow as tf
from time import gmtime, strftime
import numpy as np
np.set_printoptions(threshold=np.inf)

log_helper.initLogging('log/' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '.log')

FLAGS = None
# CHAR_NUM = 2
CHAR_NUM = 8877

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 64, 64, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Third convolutional layer -- maps 64 feature maps to 128.
  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([5, 5, 128, 256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Third pooling layer.
  with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)

  # Fourth convolutional layer -- maps 64 feature maps to 128.
  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([5, 5, 256, 512])
    b_conv4 = bias_variable([512])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

  # Fourth pooling layer.
  with tf.name_scope('pool4'):
    # h_pool4 = max_pool_2x2(h_conv4)
    h_pool4 = h_conv4

  # 5th convolutional layer -- maps 64 feature maps to 128.
  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([5, 5, 512, 512])
    b_conv5 = bias_variable([512])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

  # 5th pooling layer.
  with tf.name_scope('pool5'):
    # h_pool5 = max_pool_2x2(h_conv5)
    h_pool5 = h_conv5

  # 6th convolutional layer -- maps 64 feature maps to 128.
  with tf.name_scope('conv6'):
    W_conv6 = weight_variable([5, 5, 512, 512])
    b_conv6 = bias_variable([512])
    h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)

  # 6th pooling layer.
  with tf.name_scope('pool6'):
    h_pool6 = max_pool_2x2(h_conv6)
    # h_pool6 = h_conv6

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([4 * 4 * 512, 1024])
    b_fc1 = bias_variable([1024])

    h_pool6_flat = tf.reshape(h_pool6, [-1, 4*4*512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, CHAR_NUM])
    b_fc2 = bias_variable([CHAR_NUM])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  mnist = read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 4096])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, CHAR_NUM])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy, global_step=global_step)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  log('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(40000):
      batch = mnist.train.next_batch(400)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        log('step %d, training accuracy %g' % (i, train_accuracy))
      _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      if i % 100 == 0:
        log('loss is %g' % loss_val)

    log('test accuracy %g' % accuracy.eval(feed_dict={
        x: get_real_images(mnist.test.images), y_: dense_to_one_hot(mnist.test.labels), keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/Users/croath/Desktop/sample/data3',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
