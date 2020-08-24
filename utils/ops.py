from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def activation_summary(var):
    tf.summary.histogram(var.op.name+'/activation', var)

def variable_summary(var):
    tf.summary.histogram(var.op.name, var)

def batch_normal(input_ ,is_training=False, name="batch_norm"):
    return tf.layers.batch_normalization(input_, momentum=0.9, epsilon=1e-4, center=True,\
                                         scale=True, training=is_training, fused=True, name=name)


def conv2d(input_, 
           output_dim,
           k_h=5, k_w=5, 
           s_h=1, s_w=1, 
           stddev=0.02,
           padding='SAME', 
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1].value, output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        # variable_summary(w)
        # variable_summary(biases)
        return conv

def conv3d(input_, output_dim,
           kernel=[4,4,4], 
           strides=[1,1,1], 
           stddev=0.02,
           padding='SAME',
           name="conv3d"):
    with tf.variable_scope(name):
        k_d, k_h, k_w = kernel
        s_d, s_h, s_w = strides
        w = tf.get_variable('weight', [k_d, k_h, k_w, input_.get_shape()[-1].value, output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, s_d, s_h, s_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        # variable_summary(w)
        # variable_summary(biases)
        return conv


def deconv3d(input_, output_shape,
             kernel, 
             strides=[2,2,2,1],
             stddev=0.02,
             padding='SAME',
             name="deconv3d"):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        k_d, k_h, k_w = kernel
        w = tf.get_variable('weight', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1].value],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        o_d, o_h, o_w, o_ch = output_shape
        s_d, s_h, s_w, s_ch = strides
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=[input_.get_shape()[0].value, o_d, o_h, o_w, o_ch],
                                        strides=[1, s_d, s_h, s_w, s_ch], padding=padding, name=name)

        biases = tf.get_variable('biases', [o_ch], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        # variable_summary(w)
        # variable_summary(biases)
        return deconv

def fully_connect(input_, output_size, stddev=0.02, bias_start=0.0, name=None):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name or "Linear"):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias

def max_pool2d(inputs,
               kernel_size,
               stride=[2, 2],
               padding='VALID',
               name='maxpool'):
  """ 2D max pooling.
  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(name) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=name)
    return outputs

def dropout(inputs,
            is_training,
            name,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(name) as sc:
    outputs = tf.cond(tf.cast(is_training, tf.bool),
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs

def instance_norm(x):

    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
