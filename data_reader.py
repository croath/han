from skimage import io
from skimage import filters
import os
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

unique_label_list = []

def get_real_images(paths):
    real_images = []
    for path in paths:
        camera = io.imread(path)
        val = filters.threshold_otsu(camera)
        result = (camera < val)*1.0
        real_images.append(result)
    np_images = numpy.array(real_images)
    np_images = np_images.reshape(np_images.shape[0], np_images.shape[1] * np_images.shape[2])
    return np_images


def extract_data(path):
    images = []
    labels = []
    for sub_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, sub_dir)):
            for filename in os.listdir(os.path.join(path, sub_dir)):
                filefullname = os.path.join(path, sub_dir, filename)
                images.append(filefullname)
                labels.append(int(filename.split('_')[0][3:], 16))
    labels = numpy.array(labels, dtype=numpy.uint32)
    create_label_list(labels)
    return numpy.array(images), labels

def create_label_list(labels):
    global unique_label_list
    if len(unique_label_list) == 0:
        unique_label_list = sorted(list(set(labels)))
        with open('labels.list', 'w') as f:
            f.writelines( list( "%s\n" % item for item in unique_label_list ) )


def dense_to_one_hot(labels_dense):
    num_classes = len(unique_label_list)
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + [unique_label_list.index(x) for x in labels_dense.ravel()]] = 1
    return labels_one_hot

class DataSet(object):

  def __init__(self,
               images,
               labels,
               seed=None):
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)

    assert images.shape[0] == labels.shape[0], (
      'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def restart_epoch(self):
      self._epochs_completed = 0
      self._index_in_epoch = 0

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return get_real_images(numpy.concatenate((images_rest_part, images_new_part), axis=0)) , dense_to_one_hot(numpy.concatenate((labels_rest_part, labels_new_part), axis=0))
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return get_real_images(self._images[start:end]), dense_to_one_hot(self._labels[start:end])


def read_data_sets(train_dir, labellist_path=None, seed=None):
    global unique_label_list
    if labellist_path != None and len(unique_label_list) == 0:
        with open(labellist_path) as f:
            unique_label_list = f.read().splitlines()

    images, labels = extract_data(train_dir)
    options = dict(seed=seed)
    data = DataSet(images, labels, **options)
    return data
