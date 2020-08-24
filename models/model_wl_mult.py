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

def T_net(p, is_training, reuse=False):
    K = 3
    with tf.variable_scope('Tnet', reuse=reuse) as scope:
        p = tf.reshape(p, [-1,cfg.num_points,3,1])
        t1 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(p, output_dim=64, k_h=1, k_w=3, padding='VALID', name='t_c1'),\
                            is_training=is_training, name='t_bn1'))
        # activation_summary(t1)
        # print(t1)
        t2 = tf.nn.relu( \
                    batch_normal(\
                        conv2d(t1, output_dim=128, k_h=1, k_w=1, padding='VALID', name='t_c2'),\
                        is_training=is_training, name='t_bn2'))
        # activation_summary(t2)
        # print(t2)
        t3 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(t2, output_dim=256, k_h=1, k_w=1, padding='VALID', name='t_c3'),\
                            is_training=is_training, name='t_bn3'))
        # activation_summary(t3)
        # print(t3)
        tmax = max_pool2d(t3, [cfg.num_points, 1], padding='VALID', name='t_maxpool')
        tmax = tf.reshape(tmax, [cfg.batch_size,-1])
        # print(tmax)

        tfully = batch_normal(fully_connect(tmax, 512, name='t_f1'), is_training=is_training, name='t_bn4')
        tfully = batch_normal(fully_connect(tfully, 256, name='t_f2'), is_training=is_training, name='t_bn5')

        weights = tf.get_variable('weights', [256, 3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(tfully, weights)
        transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [cfg.batch_size, K, K])  
        return transform

def FT_net(p_feat, is_training, reuse=False):
    K = 64
    with tf.variable_scope('FTnet', reuse=reuse) as scope:
        ft1 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(p_feat, output_dim=64, k_h=1, k_w=1, padding='VALID', name='ft_c1'),\
                            is_training=is_training, name='ft_bn1'))
        # activation_summary(ft1)
        # print(ft1)
        ft2 = tf.nn.relu( \
                    batch_normal(\
                        conv2d(ft1, output_dim=128, k_h=1, k_w=1, padding='VALID', name='ft_c2'),\
                        is_training=is_training, name='ft_bn2'))
        # activation_summary(ft2)
        # print(ft2)
        ft3 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(ft2, output_dim=256, k_h=1, k_w=1, padding='VALID', name='ft_c3'),\
                            is_training=is_training, name='ft_bn3'))
        # activation_summary(ft3)
        # print(ft3)
        tmax = max_pool2d(ft3, [cfg.num_points, 1], padding='VALID', name='ft_maxpool')
        tmax = tf.reshape(tmax, [cfg.batch_size,-1])
        # print(tmax)

        ftfully = batch_normal(fully_connect(tmax, 512, name='ft_f1'), is_training=is_training, name='ft_bn4')
        ftfully = batch_normal(fully_connect(ftfully, 256, name='ft_f2'), is_training=is_training, name='ft_bn5')

        weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(ftfully, weights)
        transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [cfg.batch_size, K, K])  
        return transform

def point_net_encoder(p_vector, is_training, reuse=False):
    """
        Add droupout layers
    """
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        # Input transform network
        input_trans = T_net(p_vector, is_training=is_training, reuse=reuse)
        p_transformed = tf.matmul(p_vector, input_trans)
        p_transformed = tf.expand_dims(p_transformed, -1)
        print(p_transformed)
        enc1 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(p_transformed, output_dim=32, k_h=1, k_w=3, padding='VALID', name='e_c1'),\
                            is_training=is_training, name='e_bn1'))
        # activation_summary(enc1)
        print(enc1)
        enc2 = tf.nn.relu( \
                    batch_normal(\
                        conv2d(enc1, output_dim=64, k_h=1, k_w=1, padding='VALID', name='e_c2'),\
                        is_training=is_training, name='e_bn2'))
        # activation_summary(enc2)
        print(enc2)

        # Feature transform network
        feat_trans = FT_net(enc2, is_training=is_training, reuse=reuse)
        squeezed_enc = tf.reshape(enc2, [cfg.batch_size, cfg.num_points, 64])
        # local features
        local_feat = tf.matmul(squeezed_enc, feat_trans)
        # print(local_feat)
        local_feat = tf.expand_dims(local_feat, [2])
        enc3 = tf.nn.relu( \
                        batch_normal(\
                            conv2d(local_feat, output_dim=64, k_h=1, k_w=1, padding='VALID', name='e_c3'),\
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
                            conv2d(enc4, output_dim=256, k_h=1, k_w=1, padding='VALID', name='e_c5'),\
                            is_training=is_training, name='e_bn5'))
        # activation_summary(enc5)
        print(enc5)
        # global features
        global_feat = max_pool2d(enc5, [cfg.num_points, 1], padding='VALID', name='e_maxpool')
        print(global_feat)
        global_feat = tf.reshape(global_feat, [cfg.batch_size, -1])
        # print(global_feat)
        global_mean = fully_connect(global_feat, output_size=cfg.z_size, name='e_gfc1')
        global_log_sigma = fully_connect(global_feat, output_size=cfg.z_size, name='e_gfc2')

        # local features
        local_feat = tf.nn.relu( \
                            batch_normal(\
                                conv2d(local_feat, output_dim=32, k_h=1, k_w=1, padding='VALID', name='e_lc1'),\
                                is_training=is_training, name='e_lbn1'))
        print(local_feat)
        local_feat = tf.reshape(local_feat, [cfg.batch_size, -1])
        local_mean = fully_connect(local_feat, output_size=cfg.z_size, name='e_lfc1')
        local_log_sigma = fully_connect(local_feat, output_size=cfg.z_size, name='e_lfc2') 
        print(global_mean)
        print(local_mean)
        return global_mean, global_log_sigma, local_mean, local_log_sigma

def local_generator(local_z, is_training, reuse=False):
    strides = [2,2,2,1]
    with tf.variable_scope('local_gen', reuse=reuse) as scope:
        local_z = tf.reshape(local_z, (cfg.batch_size, 1, 1, 1, cfg.z_size))
        gen1 = tf.nn.relu(batch_normal(\
                                deconv3d(local_z, output_shape=[4,4,4,512], kernel=[4,4,4], strides=[1,1,1,1], padding='VALID', name='gl_decov1'),\
                                is_training=is_training, name='gl_bn1'))
        # activation_summary(gen1)
        gen2 = tf.nn.relu(batch_normal(\
                                deconv3d(gen1, output_shape=[8,8,8,256], kernel=[4,4,4], strides=strides, name='gl_decov2'),\
                                is_training=is_training, name='gl_bn2'))
        # activation_summary(gen2)
        gen3 = tf.nn.relu(batch_normal(\
                                deconv3d(gen2, output_shape=[16,16,16,128], kernel=[4,4,4], strides=strides, name='gl_decov3'),\
                                is_training=is_training, name='gl_bn3'))
        # activation_summary(gen3)
        gen4 = tf.nn.relu(batch_normal(\
                                deconv3d(gen3, output_shape=[32,32,32,64], kernel=[4,4,4], strides=strides, name='gl_decov4'),\
                                is_training=is_training, name='gl_bn4'))
        # activation_summary(gen4)
        gen5 = tf.nn.relu(batch_normal(\
                                deconv3d(gen4, output_shape=[64,64,64,1], kernel=[4,4,4], strides=strides, name='gl_decov5'),\
                                is_training=is_training, name='gl_bn5'))
        # activation_summary(gen5)
        return gen1, gen2, gen3, gen4, gen5

def global_generator(glob_z, loc4=None, loc8=None, loc16=None, loc32=None, loc64=None, from_pointnet=True, is_training=False, reuse=False):
    strides = [2,2,2,1]
    with tf.variable_scope('glob_gen', reuse=reuse) as scope:
        print(glob_z)
        if from_pointnet:
            glob_z = tf.reshape(glob_z, (cfg.batch_size, 1, 1, 1, cfg.z_size))
            gen1 = batch_normal(\
                                    deconv3d(glob_z, output_shape=[4,4,4,512], kernel=[4,4,4], strides=[1,1,1,1], padding='VALID', name='gg_decov1'),\
                                    is_training=is_training, name='gg_bn1')
            gen1 = tf.nn.relu(tf.multiply(gen1, loc4))
            # activation_summary(gen1)
            gen2 = batch_normal(\
                                    deconv3d(gen1, output_shape=[8,8,8,256], kernel=[4,4,4], strides=strides, name='gg_decov2'),\
                                    is_training=is_training, name='gg_bn2')
            gen2 = tf.nn.relu(tf.multiply(gen2, loc8))
            # activation_summary(gen2)
            gen3 = batch_normal(\
                                    deconv3d(gen2, output_shape=[16,16,16,128], kernel=[4,4,4], strides=strides, name='gg_decov3'),\
                                    is_training=is_training, name='gg_bn3')
            gen3 = tf.nn.relu(tf.multiply(gen3, loc16))
            # activation_summary(gen3)
            gen4 = batch_normal(\
                                    deconv3d(gen3, output_shape=[32,32,32,64], kernel=[4,4,4], strides=strides, name='gg_decov4'),\
                                    is_training=is_training, name='gg_bn4')
            gen4 = tf.nn.relu(tf.multiply(gen4, loc32))
            # activation_summary(gen4)
            gen5 = deconv3d(gen4, output_shape=[64,64,64,1], kernel=[4,4,4], strides=strides, name='gg_decov5')
            gen5 = tf.multiply(gen5, loc64)
            gen_from_point = tf.nn.tanh(gen5)
            # activation_summary(gen5)
            print(gen_from_point)
            return gen_from_point
        else:
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
            gen_from_z = tf.nn.tanh(gen5)
            # activation_summary(gen5)
            print(gen_from_z)
            return gen_from_z

# def global_generator(glob_z, loc4=0.0, loc8=0.0, loc16=0.0, loc32=0.0, loc64=0.0, from_pointnet, is_training, reuse=False):
#     strides = [2,2,2,1]
#     with tf.variable_scope('glob_gen', reuse=reuse) as scope:
#         glob_z = tf.reshape(glob_z, (cfg.batch_size, 1, 1, 1, cfg.z_size))
#         gen1 = batch_normal(\
#                                 deconv3d(glob_z, output_shape=[4,4,4,512], kernel=[4,4,4], strides=[1,1,1,1], padding='VALID', name='gg_decov1'),\
#                                 is_training=is_training, name='gg_bn1')
#         gen1 = tf.cond(from_pointnet,
#                         lambda: tf.nn.relu(tf.add(gen1, loc4)),
#                         lambda: tf.nn.relu(gen1))
#         # activation_summary(gen1)
#         gen2 = batch_normal(\
#                                 deconv3d(gen1, output_shape=[8,8,8,256], kernel=[4,4,4], strides=strides, name='gg_decov2'),\
#                                 is_training=is_training, name='gg_bn2')
#         gen2 = tf.cond(from_pointnet, tf.bool, 
#                         lambda: tf.nn.relu(tf.add(gen2, loc8)),
#                         lambda: tf.nn.relu(gen2))
#         # activation_summary(gen2)
#         gen3 = batch_normal(\
#                                 deconv3d(gen2, output_shape=[16,16,16,128], kernel=[4,4,4], strides=strides, name='gg_decov3'),\
#                                 is_training=is_training, name='gg_bn3')
#         gen3 = tf.cond(from_pointnet, 
#                         lambda: tf.nn.relu(tf.add(gen3, loc16)),
#                         lambda: tf.nn.relu(gen3))
#         # activation_summary(gen3)
#         gen4 = batch_normal(\
#                                 deconv3d(gen3, output_shape=[32,32,32,64], kernel=[4,4,4], strides=strides, name='gg_decov4'),\
#                                 is_training=is_training, name='gg_bn4')
#         gen4 = tf.cond(from_pointnet, 
#                         lambda: tf.nn.relu(tf.add(gen4, loc32)),
#                         lambda: tf.nn.relu(gen4))
#         # activation_summary(gen4)
#         gen5 = deconv3d(gen4, output_shape=[64,64,64,1], kernel=[4,4,4], strides=strides, name='gg_decov5')
#         gen5 = tf.cond(from_pointnet, 
#                         lambda: tf.nn.tanh(tf.add(gen5, loc64)),\
#                         lambda: tf.nn.tanh(gen5))
#         # activation_summary(gen5)
#         print(gen5)

#         return gen5

def discriminator(x_var, is_training, reuse=False):
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
    print('Building model with local...')
    global_mean, global_log_sigma, local_mean, local_log_sigma = point_net_encoder(p_vector, is_training=True, reuse=False)
    local_z = sample_z(local_mean, local_log_sigma)
    glob_z = sample_z(global_mean, global_log_sigma)

    loc4, loc8, loc16, loc32, loc64 = local_generator(local_z, is_training=True, reuse=False)
    s_p = global_generator(glob_z, loc4, loc8, loc16, loc32, loc64, is_training=True, reuse=False)
    s_p_sigmoid, s_p_nosig = discriminator(s_p, is_training=True, reuse=False)

    s_z = global_generator(z_vector,from_pointnet=False,is_training=True, reuse=True)
    s_z_sigmoid, s_z_nosig = discriminator(s_z,is_training=True, reuse=True)
    s_sigmoid, s_nosig = discriminator(s_vector, is_training=True, reuse=True)

    return global_mean, global_log_sigma, local_mean, local_log_sigma,\
           s_p, s_p_sigmoid, s_p_nosig, s_z, s_z_nosig, s_z_sigmoid, s_sigmoid, s_nosig


def vae_gan_loss(global_mean, global_log_sigma, local_mean, local_log_sigma, s_vector, s_p, s_p_nosig, s_z_nosig, s_nosig):
    # D_loss
    d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s_nosig), logits=s_nosig)
    d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(s_z_nosig), logits=s_z_nosig)
    # d_loss = d_real_loss + d_fake_loss
    # 3D-VAE-GAN does not use loss from encoder but in Autoencoding beyond pixels paper does.
    d_encode_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(s_p_nosig), logits=s_p_nosig)
    d_loss = tf.reduce_mean(d_real_loss + d_fake_loss + d_encode_loss)

    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_d_real_loss = tf.summary.scalar("d_real_loss", tf.reduce_mean(d_real_loss))
    sum_d_fake_loss = tf.summary.scalar("d_fake_loss", tf.reduce_mean(d_fake_loss))
    sum_d_encode_loss = tf.summary.scalar("d_encode_loss", tf.reduce_mean(d_encode_loss))

    # reconstruction loss)
    # rec_loss = tf.sqrt(tf.reduce_sum(tf.squared_difference(s_p, s_vector), axis=[1,2,3,4])) # L2 - Euclidean Distance
    # rec_loss = tf.nn.l2_loss((s_p, s_vector)) # output = sum(t^2) / 2
    # rec_loss = tf.abs(s_p - s_vector)
    # rec_loss = tf.losses.mean_squared_error(s_vector, s_p)
    rec_loss = tf.losses.absolute_difference(s_vector, s_p)
    sum_rec_loss = tf.summary.scalar("rec_loss", tf.reduce_mean(rec_loss))
    
    # KL loss
    local_kl_loss = KL_loss(local_mean, local_log_sigma)
    sum_local_kl_loss = tf.summary.scalar("Local_KL_loss", tf.reduce_mean(local_kl_loss))
    glob_kl_loss = KL_loss(global_mean, global_log_sigma)
    sum_glob_kl_loss = tf.summary.scalar("Glob_KL_loss", tf.reduce_mean(glob_kl_loss))
    total_kl_loss = local_kl_loss + glob_kl_loss

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
           tf.reduce_mean(rec_loss), tf.reduce_mean(local_kl_loss), tf.reduce_mean(glob_kl_loss), tf.reduce_mean(total_kl_loss),\
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

# def KL_loss(z_mean, z_sigma):
#     # mean = tf.clip_by_value(mean, -10.0, 10.0)
#     # var = tf.clip_by_value(var, -10.0, 10.0)
#     # shape = z_sigma.get_shape().as_list()
#     # I = tf.eye(z_size)
#     return tf.reduce_mean(tf.distributions.kl_divergence(\
#                                 tf.distributions.Normal(loc=z_mean, scale=z_sigma),\
#                                 tf.distributions.Normal(loc=0.0, scale=np.ones(z_size, dtype=np.float32))))
