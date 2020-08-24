from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(sys.path[0])
sys.path.append(os.path.join(sys.path[0], '../utils'))

from ops import *
from config import cfg

def point_net_encoder(p_vector, is_training, reuse=False):
    """
        Add droupout layers
    """
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # Input transform network
        p_transformed = tf.expand_dims(p_vector, -1)
        print(p_transformed)
        enc1 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(p_transformed, output_dim=1024, k_h=1, k_w=3, padding='VALID', name='e_c1'),\
                            is_training=is_training, name='e_bn1'))
        # activation_summary(enc1)
        print(enc1)
        enc2 = tf.nn.relu( \
                    batch_normal(\
                        conv2d(enc1, output_dim=512, k_h=1, k_w=1, padding='VALID', name='e_c2'),\
                        is_training=is_training, name='e_bn2'))
        # activation_summary(enc2)
        print(enc2)
        enc3 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(enc2, output_dim=256, k_h=1, k_w=1, padding='VALID', name='e_c3'),\
                            is_training=is_training, name='e_bn3'))
        # activation_summary(enc3)
        print(enc3)
        enc3 = dropout(enc3, keep_prob=0.7, is_training=is_training, name='e_dp1')
        enc4 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(enc3, output_dim=128, k_h=1, k_w=1, padding='VALID', name='e_c4'),\
                            is_training=is_training, name='e_bn4'))
        # activation_summary(enc4)
        print(enc4)
        enc4 = dropout(enc4, keep_prob=0.7, is_training=is_training, name='e_dp2')
        enc5 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(enc4, output_dim=64, k_h=1, k_w=1, padding='VALID', name='e_c5'),\
                            is_training=is_training, name='e_bn5'))
        # activation_summary(enc5)
        print(enc5)
        enc6 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(enc5, output_dim=32, k_h=1, k_w=1, padding='VALID', name='e_c6'),\
                            is_training=is_training, name='e_bn6'))
        # activation_summary(enc5)
        print(enc6)
        enc7 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(enc6, output_dim=16, k_h=1, k_w=1, padding='VALID', name='e_c7'),\
                            is_training=is_training, name='e_bn7'))
        # activation_summary(enc5)
        print(enc7)
        # global features
        global_feat = max_pool2d(enc7, [cfg.num_points, 1], padding='VALID', name='e_maxpool')
        print(global_feat)
        global_feat = tf.reshape(global_feat, [cfg.batch_size, -1])
        # print(global_feat)
        global_mean = fully_connect(global_feat, output_size=cfg.z_size, name='e_gfc1')
        global_log_sigma = fully_connect(global_feat, output_size=cfg.z_size, name='e_gfc2')

        return global_mean, global_log_sigma

def global_generator(glob_z, is_training=False, reuse=False):
    strides = [2,2,2,1]
    with tf.variable_scope('glob_gen', reuse=reuse) as scope:
        glob_z = tf.reshape(glob_z, (cfg.batch_size, 1, 1, 1, cfg.z_size))
        gen1 = tf.nn.relu(batch_normal(\
                                deconv3d(glob_z, output_shape=[4,4,4,512], kernel=[4,4,4], strides=[1,1,1,1], padding='VALID', name='gg_decov1'),\
                                is_training=is_training, name='gg_bn1'))
        # activation_summary(gen1)
        gen2 = tf.nn.relu(batch_normal(\
                                deconv3d(gen1, output_shape=[8,8,8,256], kernel=[4,4,4], strides=strides, name='gg_decov2'),\
                                is_training=is_training, name='gg_bn2'))
        # activation_summary(gen2)
        gen3 = tf.nn.relu(batch_normal(\
                                deconv3d(gen2, output_shape=[16,16,16,128], kernel=[4,4,4], strides=strides, name='gg_decov3'),\
                                is_training=is_training, name='gg_bn3'))
        # activation_summary(gen3)
        gen4 = tf.nn.relu(batch_normal(\
                                deconv3d(gen3, output_shape=[32,32,32,64], kernel=[4,4,4], strides=strides, name='gg_decov4'),\
                                is_training=is_training, name='gg_bn4'))
        # activation_summary(gen4)
        gen5 = deconv3d(gen4, output_shape=[64,64,64,1], kernel=[4,4,4], strides=strides, name='gg_decov5')
        gen5 = tf.nn.tanh(gen5)
        # activation_summary(gen5)
        print(gen5)
        return gen5

def discriminator(x_var, is_training=False, reuse=False):
    strides = [2,2,2]
    with tf.variable_scope('disc', reuse=reuse) as scope:
        dis1 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(x_var, output_dim=64, strides=strides, padding='SAME', name='d_conv1'),
                                    is_training=is_training, name='d_bn1'), cfg.leak_value)
        # activation_summary(dis1)
        dis2 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(dis1, output_dim=32, kernel=[1,1,1], padding='SAME', name='d_conv2'),
                                    is_training=is_training, name='d_bn2'), cfg.leak_value)
        # activation_summary(dis2)
        dis3 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(dis2, output_dim=128, strides=strides, padding='SAME', name='d_conv3'),
                                    is_training=is_training, name='d_bn3'), cfg.leak_value)
        # activation_summary(dis3)
        dp = dropout(dis3, is_training=is_training, keep_prob=0.7, name='d_dp1')
        dis4 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(dp, output_dim=256, strides=strides, padding='SAME', name='d_conv4'),
                                    is_training=is_training, name='d_bn4'), cfg.leak_value)
        # activation_summary(dis4)
        dis5 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(dis4, output_dim=128, kernel=[1,1,1], padding='SAME', name='d_conv5'),
                                    is_training=is_training, name='d_bn5'), cfg.leak_value)
        # activation_summary(dis2)
        dis6 = tf.nn.leaky_relu(batch_normal(\
                                    conv3d(dis5, output_dim=512, strides=strides, padding='SAME', name='d_conv6'),
                                    is_training=is_training, name='d_bn6'), cfg.leak_value)
        dis7 = conv3d(dis6, output_dim=1, strides=[1,1,1], padding='VALID', name='d_conv7')
        # activation_summary(dis7)
        dis7_sigmoid = tf.nn.sigmoid(dis7)
        # activation_summary(dis7_sigmoid)
        print(dis7)
        print(dis7_sigmoid)
        return dis7_sigmoid, dis7

#-----------------------------------------------------------------------------#
def build_model(p_vector, s_vector, z_vector):
    # Encoder
    print('Building model baseline...')
    global_mean, global_log_sigma = point_net_encoder(p_vector, is_training=True, reuse=False)
    glob_z = sample_z(global_mean, global_log_sigma)

    s_p = global_generator(glob_z,is_training=True, reuse=False)
    s_p_sigmoid, s_p_nosig = discriminator(s_p, is_training=True, reuse=False)

    s_z = global_generator(z_vector,is_training=True, reuse=True)
    s_z_sigmoid, s_z_nosig = discriminator(s_z,is_training=True, reuse=True)
    s_sigmoid, s_nosig = discriminator(s_vector, is_training=True, reuse=True)

    return global_mean, global_log_sigma, s_p, s_p_sigmoid, s_p_nosig, \
           s_z, s_z_nosig, s_z_sigmoid, s_sigmoid, s_nosig


def vae_gan_loss(global_mean, global_log_sigma, s_vector, s_p, s_p_nosig, s_z_nosig, s_nosig):
    # D_loss
    d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s_nosig), logits=s_nosig)
    d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(s_z_nosig), logits=s_z_nosig)
    d_encode_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(s_p_nosig), logits=s_p_nosig)
    d_loss = tf.reduce_mean(d_real_loss + d_fake_loss + d_encode_loss)

    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_d_real_loss = tf.summary.scalar("d_real_loss", tf.reduce_mean(d_real_loss))
    sum_d_fake_loss = tf.summary.scalar("d_fake_loss", tf.reduce_mean(d_fake_loss))
    sum_d_encode_loss = tf.summary.scalar("d_encode_loss", tf.reduce_mean(d_encode_loss))

    rec_loss = tf.abs(s_p - s_vector)
    sum_rec_loss = tf.summary.scalar("rec_loss", tf.reduce_mean(rec_loss))
    
    # KL loss
    glob_kl_loss = KL_loss(global_mean, global_log_sigma)
    sum_glob_kl_loss = tf.summary.scalar("Glob_KL_loss", tf.reduce_mean(glob_kl_loss))
    total_kl_loss = glob_kl_loss

    # E_loss
    e_loss = tf.reduce_mean(cfg.alpha1 * total_kl_loss + cfg.alpha2 * rec_loss)
    sum_e_loss = tf.summary.scalar("e_loss", e_loss)

    # G_loss
    g_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s_z_nosig), logits=s_z_nosig)
    g_encode_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s_p_nosig), logits=s_p_nosig)
    # g_loss = g_fake_loss + rec_loss
    g_loss = tf.reduce_mean(g_fake_loss + g_encode_loss + cfg.alpha2 * rec_loss)

    sum_g_fake_loss = tf.summary.scalar("g_fake_loss", tf.reduce_mean(g_fake_loss))
    sum_g_encode_loss = tf.summary.scalar("g_en_loss", tf.reduce_mean(g_encode_loss))
    sum_g_loss = tf.summary.scalar("g_loss", g_loss)

    return tf.reduce_mean(d_real_loss), tf.reduce_mean(d_fake_loss), tf.reduce_mean(d_encode_loss), d_loss,\
           tf.reduce_mean(rec_loss), tf.reduce_mean(glob_kl_loss), tf.reduce_mean(total_kl_loss),\
           e_loss, tf.reduce_mean(g_fake_loss), tf.reduce_mean(g_encode_loss), g_loss


#-----------------------------------------------------------------------------#
def sample_z(mu, log_sigma):
    # eps = tf.random_uniform(shape=(cfg.batch_size, cfg.z_size), minval=-1.0, maxval=1.0)
    eps = tf.random_normal(shape=(cfg.batch_size, cfg.z_size))
    return mu + tf.exp(log_sigma / 2) * eps

def KL_loss(z_mean, z_log_sigma):
    # z_mean = tf.clip_by_value(z_mean, -10.0, 10.0)
    # z_log_sigma = tf.clip_by_value(z_log_sigma, -10.0, 10.0)
    # return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_sigma) - tf.log(tf.square(z_sigma)) - 1,1))
    return -0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=-1)

