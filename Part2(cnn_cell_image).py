"""Assignment 3 : CSCI 964
    Nirmal Gajera (5626924)
    Part 2 (Deep Learning) (CNN cell Image)"""

import sys
import cv2
import os
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import numpy as np
import csv
from random import randint
FLAGS = None


def convert_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def load_labels_from_csv(filename):
  labels = []
  with open(filename, 'r') as csvfile:
    data = csv.reader(csvfile)
    for row in data:
      if(row[1] == "Homogeneous"):
        labels.append(1)
      if(row[1] == "Centromere"):
        labels.append(2)
      if(row[1] == "Golgi"):
        labels.append(3)
      if(row[1] == "Nucleolar"):
        labels.append(4)
      if(row[1] == "NuMem"):
        labels.append(5)
      if(row[1] == "Speckled"):
        labels.append(6)
  return np.array(labels)

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def load_images_from_folder(folder, alllabels):
  images = []
  labels = []
  for filename in os.listdir(folder):
      img = cv2.imread(os.path.join(folder,filename), 0)
      if img is not None:
        newimage = cv2.resize(img,(64,64))
        images.append(np.array(newimage).flatten()/255.0)
        labels.append(alllabels[int(filename.split(".")[0]) - 1])
        newimage = rotateImage(newimage, randint(1,89))
        images.append(np.array(newimage).flatten()/255.0)
        labels.append(alllabels[int(filename.split(".")[0]) - 1])
  return np.array(images), np.array(labels)

class DataSet(object):
  def __init__(self,
               images,
               labels):
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

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    if self._epochs_completed == 0 and start == 0:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
   
    if start + batch_size > self._num_examples:      
      self._epochs_completed += 1     
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def read_data_sets():
  labels = load_labels_from_csv("gt_training.csv")
  train, train_labels = load_images_from_folder("training", labels)
  train_labels = convert_to_one_hot(train_labels, 6)

  test, test_labels = load_images_from_folder("test", labels)
  test_labels = convert_to_one_hot(test_labels, 6)
  validation, validation_labels = load_images_from_folder("validation", labels)
  validation_labels = convert_to_one_hot(validation_labels, 6)

  train = DataSet(
      train, train_labels)
  validation = DataSet(
      validation,
      validation_labels)
  test = DataSet(
      test, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)



def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  data = read_data_sets()
  x = tf.placeholder(tf.float32, [None, 4096])
  W = tf.Variable(tf.zeros([4096, 6]))
  b = tf.Variable(tf.zeros([6]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, 6])
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  x_image = tf.reshape(x, [-1,64,64,1])
  
  W_conv1 = tf.truncated_normal([5, 5, 1, 8], stddev=0.1)
  b_conv1 = bias_variable([8])
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  print(h_pool1)
  
  W_conv2 = tf.truncated_normal([5, 5, 8, 16], stddev=0.1)
  b_conv2 = bias_variable([16]) 
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,strides=[1, 1, 1, 1], padding='SAME')  + b_conv2)
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  print(h_pool2)

  W_conv3 = tf.truncated_normal([5, 5, 16, 32], stddev=0.1)
  b_conv3 = bias_variable([32]) 
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3,strides=[1, 1, 1, 1], padding='SAME')  + b_conv3)
  h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  print(h_pool3)

  
  W_fc1 = tf.truncated_normal([8 * 8 * 32, 1024], stddev=0.1)
  b_fc1 = bias_variable([1024]) 

  h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 32])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  

  
  W_fc2 = tf.truncated_normal([1024, 6], stddev=0.1)
  b_fc2 = bias_variable([6]) 

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.global_variables_initializer())

  for i in range(100):
    batch = data.train.next_batch(200)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
      if i%1000 == 0:
        print("validation accuracy %g"%accuracy.eval(feed_dict={
          x: data.validation.images, y_: data.validation.labels, keep_prob: 1.0}))
        print("test accuracy %g"%accuracy.eval(feed_dict={
          x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) 
  print("validation accuracy %g"%accuracy.eval(feed_dict={x: data.validation.images, y_: data.validation.labels, keep_prob: 1.0}))
  print("test accuracy %g"%accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main)
