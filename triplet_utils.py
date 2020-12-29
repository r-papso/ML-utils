import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, batch_size, rnd_class_number, x, y):
    if batch_size % rnd_class_number != 0:
      raise ValueError('batch_size has to be divisible by rnd_class_number')

    self._batch_size = batch_size
    self._rnd_class_number = rnd_class_number
    self._directory = self.__generate_class_directory(x, y)

  def __getitem__(self, idx):
    data_shape = self._directory[0][0].shape
    batch_x = np.empty((self._batch_size, data_shape[0], data_shape[1], data_shape[2]))
    batch_y = np.empty(self._batch_size)
    data_per_class = self._batch_size // self._rnd_class_number
    counter = 0

    for i in random.sample(range(len(self._directory)), self._rnd_class_number):
      for j in random.sample(range(len(self._directory[i])), data_per_class):
        batch_x[counter] = self._directory[i][j]
        batch_y[counter] = i
        counter += 1

    return batch_x, batch_y

  def __len__(self):
    return sum([len(x) for x in self._directory]) // self._batch_size

  def __generate_class_directory(self, x, y):
    directory = [[] for i in range(np.unique(y).shape[0])]
    [directory[y[i]].append(x[i]) for i in range(x.shape[0])]

    return directory
    
    
class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch > 0 and epoch % 20 == 0:
      k_nearest.assign_add(1)
      print("Epoch {} k_nearest set to: {}".format(epoch, k_nearest))
    

def fashion_mnist_model(input_shape, model_type='reference', normalize=None, minmax_a=0, minmax_b=1):
  if model_type not in ['reference', 'triplet']:
    raise ValueError('Invalid input type: {}'.format(model_type))

  if normalize is not None and normalize not in ['l2', 'minmax']:
    raise ValueError('Invalid input type: {}'.format(normalize))

  model_input = keras.Input(shape=input_shape)

  x = layers.Conv2D(16, 3, padding='same', activation='relu')(model_input)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPool2D()(x)
  x = layers.Dropout(0.15)(x)
  x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPool2D()(x)
  x = layers.Dropout(0.15)(x)
  x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(10)(x)

  if model_type == 'reference':
    x = layers.Softmax()(x)
  elif model_type == 'triplet' and normalize == 'l2':
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
  elif model_type == 'triplet' and normalize == 'minmax':
    x = layers.Lambda(lambda x: minmax_normalize(x, minmax_a, minmax_b))(x)

  model = keras.Model(model_input, x)
  return model


def cifar_model(input_shape, model_type='reference', normalize=None, minmax_a=0, minmax_b=1):
  if model_type not in ['reference', 'triplet']:
    raise ValueError('Invalid input type: {}'.format(model_type))

  if normalize is not None and normalize not in ['l2', 'minmax']:
    raise ValueError('Invalid input type: {}'.format(normalize))

  model_input = keras.Input(shape=input_shape)

  x = layers.Conv2D(32, 3, padding='same', activation='elu')(model_input)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(32, 3, padding='valid', activation='elu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(32, 3, padding='valid', activation='elu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(32, 3, padding='valid', activation='elu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(32, 3, padding='same', activation='elu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPool2D()(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(64, 3, padding='same', activation='elu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(64, 3, padding='same', activation='elu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPool2D()(x)
  x = layers.Dropout(0.3)(x)

  x = layers.Conv2D(128, 3, padding='same', activation='elu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(128, 3, padding='same', activation='elu')(x)
  x = layers.BatchNormalization()(x)

  x = layers.MaxPool2D()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(10)(x)

  if model_type == 'reference':
    x = layers.Softmax()(x)
  elif model_type == 'triplet' and normalize == 'l2':
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
  elif model_type == 'triplet' and normalize == 'minmax':
    x = layers.Lambda(lambda x: minmax_normalize(x, minmax_a, minmax_b))(x)

  model = keras.Model(model_input, x)
  return model

  
def k_nearest_triplet(k, alpha=0.5, beta=1.5, margin_type=0, greater_negatives=True):

  def loss(y_true, y_pred):
    labels, embeddings = y_true, y_pred
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Compute distance matrix and sort it's rows
    r = tf.reshape(tf.reduce_sum(embeddings * embeddings, 1), [-1, 1])
    D = r - 2 * tf.matmul(embeddings, embeddings, transpose_b=True) + tf.transpose(r)
    D_sorted = tf.sort(D)

    # Get adjacency matrix and sort it's rows by D_sorted
    adjacency = tf.gather(tf.equal(labels, tf.transpose(labels)), tf.argsort(D), batch_dims=1)
    adjacency_not = tf.logical_not(adjacency)

    # Get positive pairs
    data_per_class = tf.unique_with_counts(labels[:,0]).count[0]
    p_indicies = tf.where(adjacency)[k::data_per_class,:]

    x_p = tf.gather_nd(D_sorted, p_indicies)

    # Get negative pairs
    tf_negatives = tf.constant(greater_negatives, dtype=tf.bool)

    mask_greater = tf.greater(D_sorted, tf.cond(tf_negatives, lambda: tf.reshape(x_p, [-1, 1]), lambda : 0.0))
    mask_greater_and_not = tf.logical_and(mask_greater, adjacency_not)
    zero_rows_mask = tf.reshape(tf.equal(tf.reduce_sum(tf.cast(mask_greater_and_not, tf.float32), axis=1), 0), [-1, 1])
    mask_n = tf.where(zero_rows_mask, adjacency_not, mask_greater_and_not)

    non_unique_indicies = tf.where(mask_n)
    x_count = tf.cumsum(tf.ones_like(non_unique_indicies[:,0])) - 1
    unique, unique_id = tf.unique(non_unique_indicies[:,0])
    unique_first = tf.math.unsorted_segment_min(x_count, unique_id, tf.shape(unique)[0])
    n_indicies = tf.gather(non_unique_indicies, unique_first)

    x_n = tf.gather_nd(D_sorted, n_indicies)

    # Compute loss
    tf_alpha = tf.constant(alpha, dtype=embeddings.dtype)
    tf_beta = tf.constant(beta, dtype=embeddings.dtype)
    tf_margin_type = tf.constant(margin_type, dtype=tf.int32)

    total_loss = tf.switch_case(tf_margin_type, branch_fns={0: lambda: hinge_margin(x_p, x_n, tf_alpha), 
                                                            1: lambda: softplus_margin(x_p, x_n), 
                                                            2: lambda: scale_margin(x_p, x_n, tf_beta),
                                                            3: lambda: hybrid_margin(x_p, x_n, tf_alpha, tf_beta)})

    final_loss = tf.truediv(tf.reduce_sum(total_loss), tf.cast(tf.shape(labels)[0], tf.float32))

    return final_loss

  return loss


def surrounding_triplet(eps, alpha=0.5, beta=1.5, margin_type=0, greater_negatives=True):

  def loss(y_true, y_pred):
    labels, embeddings = y_true, y_pred
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Compute distance matrix and sort it's rows
    r = tf.reshape(tf.reduce_sum(embeddings * embeddings, 1), [-1, 1])
    D = r - 2 * tf.matmul(embeddings, embeddings, transpose_b=True) + tf.transpose(r)
    D_sorted = tf.sort(D)

    # Get adjacency matrix and sort it's rows by D_sorted
    adjacency = tf.gather(tf.equal(labels, tf.transpose(labels)), tf.argsort(D), batch_dims=1)
    adjacency_not = tf.logical_not(adjacency)

    # Get positive pairs
    mask_p = tf.logical_and(adjacency, tf.less(D_sorted, eps))

    non_unique_idxs = tf.where(mask_p)
    x_count = tf.cumsum(tf.ones_like(non_unique_idxs[:,0])) - 1
    unique, unique_id = tf.unique(non_unique_idxs[:,0])
    unique_max = tf.math.unsorted_segment_max(x_count, unique_id, tf.shape(unique)[0])
    p_indicies = tf.gather(non_unique_idxs, unique_max)

    x_p = tf.gather_nd(D_sorted, p_indicies)

    # Get negative pairs
    tf_negatives = tf.constant(greater_negatives, dtype=tf.bool)

    mask_greater = tf.greater(D_sorted, tf.cond(tf_negatives, lambda: tf.reshape(x_p, [-1, 1]), lambda : 0.0))
    mask_n = tf.logical_and(mask_greater, adjacency_not)

    non_unique_idxs = tf.where(mask_n)
    x_count = tf.cumsum(tf.ones_like(non_unique_idxs[:,0])) - 1
    unique, unique_id = tf.unique(non_unique_idxs[:,0])
    unique_min = tf.math.unsorted_segment_min(x_count, unique_id, tf.shape(unique)[0])
    n_indicies = tf.gather(non_unique_idxs, unique_min)

    x_n = tf.gather_nd(D_sorted, n_indicies)

    # Compute loss
    tf_alpha = tf.constant(alpha, dtype=embeddings.dtype)
    tf_beta = tf.constant(beta, dtype=embeddings.dtype)
    tf_margin_type = tf.constant(margin_type, dtype=tf.int32)

    total_loss = tf.switch_case(tf_margin_type, branch_fns={0: lambda: hinge_margin(x_p, x_n, tf_alpha), 
                                                            1: lambda: softplus_margin(x_p, x_n), 
                                                            2: lambda: scale_margin(x_p, x_n, tf_beta),
                                                            3: lambda: hybrid_margin(x_p, x_n, tf_alpha, tf_beta)})
    
    final_loss = tf.truediv(tf.reduce_sum(total_loss), tf.cast(tf.shape(labels)[0], tf.float32))

    return final_loss

  return loss


def delta_triplet(alpha=0.5, beta=1.5, margin_type=0, greater_negatives=True):

  def loss(y_true, y_pred):
    labels, embeddings = y_true, y_pred
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Compute distance matrix and sort it's rows
    r = tf.reshape(tf.reduce_sum(embeddings * embeddings, 1), [-1, 1])
    D = r - 2 * tf.matmul(embeddings, embeddings, transpose_b=True) + tf.transpose(r)
    D_sorted = tf.sort(D)

    # Get adjacency matrix and sort it's rows by D_sorted
    adjacency = tf.gather(tf.equal(labels, tf.transpose(labels)), tf.argsort(D), batch_dims=1)
    adjacency_not = tf.logical_not(adjacency)

    # Get distances between positive pairs only
    p_dists = tf.reshape(tf.gather_nd(D_sorted, tf.where(adjacency)), [tf.shape(D)[0], -1])
    p_dist_sliced = tf.slice(p_dists, [0, 1], [tf.shape(p_dists)[0], tf.shape(p_dists)[1] - 1])

    # Get positive pairs
    diff = p_dist_sliced[:,1:] - p_dist_sliced[:,:-1]
    diff_mean = tf.reshape(tf.reduce_mean(diff, axis=1), [-1, 1])

    non_unique_idxs = tf.where(tf.greater(diff, diff_mean))
    x_count = tf.cumsum(tf.ones_like(non_unique_idxs[:,0])) - 1
    unique, unique_id = tf.unique(non_unique_idxs[:,0])
    unique_min = tf.math.unsorted_segment_min(x_count, unique_id, tf.shape(unique)[0])
    p_indicies = tf.gather(non_unique_idxs, unique_min)
    
    x_p = tf.gather_nd(p_dist_sliced, p_indicies)

    # Get negative pairs
    tf_negatives = tf.constant(greater_negatives, dtype=tf.bool)

    mask_greater = tf.greater(D_sorted, tf.cond(tf_negatives, lambda: tf.reshape(x_p, [-1, 1]), lambda : 0.0))
    mask_greater_and_not = tf.logical_and(mask_greater, adjacency_not)
    zero_rows_mask = tf.reshape(tf.equal(tf.reduce_sum(tf.cast(mask_greater_and_not, tf.float32), axis=1), 0), [-1, 1])
    mask_n = tf.where(zero_rows_mask, adjacency_not, mask_greater_and_not)

    non_unique_indicies = tf.where(mask_n)
    x_count = tf.cumsum(tf.ones_like(non_unique_indicies[:,0])) - 1
    unique, unique_id = tf.unique(non_unique_indicies[:,0])
    unique_first = tf.math.unsorted_segment_min(x_count, unique_id, tf.shape(unique)[0])
    n_indicies = tf.gather(non_unique_indicies, unique_first)

    x_n = tf.gather_nd(D_sorted, n_indicies)

    # Compute loss
    tf_alpha = tf.constant(alpha, dtype=embeddings.dtype)
    tf_beta = tf.constant(beta, dtype=embeddings.dtype)
    tf_margin_type = tf.constant(margin_type, dtype=tf.int32)

    total_loss = tf.switch_case(tf_margin_type, branch_fns={0: lambda: hinge_margin(x_p, x_n, tf_alpha), 
                                                            1: lambda: softplus_margin(x_p, x_n), 
                                                            2: lambda: scale_margin(x_p, x_n, tf_beta),
                                                            3: lambda: hybrid_margin(x_p, x_n, tf_alpha, tf_beta)})
    
    final_loss = tf.truediv(tf.reduce_sum(total_loss), tf.cast(tf.shape(labels)[0], tf.float32))

    return final_loss

  return loss


def hinge_margin(x_p, x_n, alpha):
  return tf.maximum(tf.add(x_p - x_n, alpha), 0.0)


def softplus_margin(x_p, x_n):
  return tf.math.softplus(x_p - x_n)


def scale_margin(x_p, x_n, beta):
  return tf.maximum(tf.subtract(tf.multiply(x_p, beta), x_n), 0.0)


def hybrid_margin(x_p, x_n, alpha, beta):
  return tf.maximum(hinge_margin(x_p, x_n, alpha), scale_margin(x_p, x_n, beta))


def minmax_normalize(x, a, b):
  tf_min = tf.reshape(tf.reduce_min(x, axis=1), [-1, 1])
  tf_max = tf.reshape(tf.reduce_max(x, axis=1), [-1, 1])
  return (b - a) * ((x - tf_min) / (tf_max - tf_min)) + a


def random_class_merge(y_train, y_test, out_file):
  y_train_merged = np.zeros(y_train.shape, dtype=np.uint8)
  y_test_merged = np.zeros(y_test.shape, dtype=np.uint8)

  classes = np.unique(y_train)
  np.random.shuffle(classes)
  class_dict = dict()

  for i in range(classes.shape[0]):
    class_dict[classes[i]] = i % (classes.shape[0] // 2)

  with open(out_file, 'w') as mapping:
    for k, v in class_dict.items():
      mapping.write('{},{}\n'.format(k, v))

  for i in range(y_train.shape[0]):
    y_train_merged[i] = class_dict[y_train[i]]

  for i in range(y_test.shape[0]):
    y_test_merged[i] = class_dict[y_test[i]]

  return y_train_merged, y_test_merged
  
  
def reassign_classes(y_train, y_test, mapping_path):
  y_train_mapped = np.zeros(y_train.shape, dtype=np.uint8)
  y_test_mapped = np.zeros(y_test.shape, dtype=np.uint8)
  class_dict = dict()

  with open(mapping_path, 'r') as mapping:
    for line in mapping:
      class_dict[int(line.split(',')[0])] = int(line.split(',')[1])

  for i in range(y_train.shape[0]):
    y_train_mapped[i] = class_dict[y_train[i]]

  for i in range(y_test.shape[0]):
    y_test_mapped[i] = class_dict[y_test[i]]

  return y_train_mapped, y_test_mapped
