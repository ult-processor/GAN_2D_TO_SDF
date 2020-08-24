from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import importlib
import os
import sys
import h5py

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(sys.path[0])
sys.path.append(os.path.join(sys.path[0], 'models'))
sys.path.append(os.path.join(sys.path[0], 'utils'))

from utils import *
from model_no_local import * 
from config import cfg
 
sdf_directory = './logs_no_local_table/train_samples/org/'
ply_directory = './logs_no_local_table/train_samples/ply/'
sp_directory = './logs_no_local_table/train_samples/sp/'
sz_directory = './logs_no_local_table/train_samples/sz/'
ckpt_directory = './logs_no_local_table/checkpoints/'

input_dir = os.path.join(sys.path[0], 'data/')
train_file = os.path.join(input_dir, "table_s64p1024_train.h5")
# test_file = os.path.join(input_dir,"chair_s64p1024_test.h5")


def trainGAN():
    global_step = tf.Variable(0, name='global_step', trainable=False) #tf.get_variable('global_step', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32)

    p_vector = tf.placeholder(shape=[cfg.batch_size,cfg.num_points,3],dtype=tf.float32)
    z_vector = tf.placeholder(shape=[cfg.batch_size,cfg.z_size],dtype=tf.float32)
    s_vector = tf.placeholder(shape=[cfg.batch_size,cfg.cube_len,cfg.cube_len,cfg.cube_len,cfg.cube_channel],dtype=tf.float32)

    #-------------------Building a model-------------------------#
    global_mean, global_log_sigma, s_p, s_p_sigmoid, s_p_nosig, \
    s_z, s_z_nosig, s_z_sigmoid, s_sigmoid, s_nosig = build_model(p_vector, s_vector, z_vector)

    #-------------------END of Building a model-------------------------#
    d_real_loss, d_fake_loss, d_encode_loss, d_loss,\
    rec_loss, glob_kl_loss, total_kl_loss, e_loss, \
    g_fake_loss, g_encode_loss, g_loss = vae_gan_loss(global_mean, global_log_sigma, s_vector, s_p,\
                                                      s_p_nosig, s_z_nosig, s_nosig)

    #-------------------END of Building a model-------------------------#

    D_s = tf.reduce_sum(tf.cast(s_sigmoid > 0.5, tf.int32))
    D_G_s_z = tf.reduce_sum(tf.cast(s_z_sigmoid <= 0.5, tf.int32))
    D_G_s_p = tf.reduce_sum(tf.cast(s_p_sigmoid <= 0.5, tf.int32))
    d_accuracy = tf.divide(D_s + D_G_s_z + D_G_s_p, 3*np.prod(s_sigmoid.shape.as_list()))

    sum_D_s = tf.summary.scalar("D_s", D_s)
    sum_D_G_s_z = tf.summary.scalar("D_G_s_z", D_G_s_z)
    sum_D_G_s_p = tf.summary.scalar("D_G_s_p", D_G_s_p)
    sum_d_accuracy = tf.summary.scalar("d_accuracy", d_accuracy)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if any(name in var.name for name in ['disc', 'd_'])]
    g_vars = [var for var in t_vars if any(name in var.name for name in ['glob_gen', 'gg_'])]
    e_vars = [var for var in t_vars if any(name in var.name for name in ['encoder', 'e_', 'Tnet', 't_', 'FTnet', 'ft_'])]

    # For ``tf.layers.batch_normalization`` updating
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_lr = tf.train.exponential_decay(cfg.init_d_lr, global_step, cfg.DECAY_STEP, cfg.D_DECAY_RATE, staircase=True)
        g_lr = tf.train.exponential_decay(cfg.init_g_lr, global_step, cfg.DECAY_STEP, cfg.G_DECAY_RATE, staircase=True)
        e_lr = tf.train.exponential_decay(cfg.init_e_lr, global_step, cfg.DECAY_STEP, cfg.E_DECAY_RATE, staircase=True)
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=cfg.beta).minimize(d_loss,var_list=d_vars)
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=cfg.beta).minimize(g_loss,var_list=g_vars)
        optimizer_op_e = tf.train.AdamOptimizer(learning_rate=e_lr,beta1=cfg.beta).minimize(e_loss,var_list=e_vars)
        # optimizer_op_d = tf.train.AdamOptimizer(learning_rate=cfg.init_d_lr,beta1=cfg.beta).minimize(d_loss,var_list=d_vars)
        # optimizer_op_g = tf.train.AdamOptimizer(learning_rate=cfg.init_g_lr,beta1=cfg.beta).minimize(g_loss,var_list=g_vars)
        # optimizer_op_e = tf.train.AdamOptimizer(learning_rate=cfg.init_e_lr,beta1=cfg.beta).minimize(e_loss,var_list=e_vars)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    ### Constraint GPU memory usage
    # [ref] https://stackoverflow.com/questions/34199233
    # [ref] https://indico.io/blog/the-good-bad-ugly-of-tensorflow/
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    summary = tf.summary.merge_all()

    with tf.Session(config=config) as sess:

        sess.run(init)
        summary_writer = tf.summary.FileWriter(ckpt_directory, sess.graph)
        flog = open(os.path.join(ckpt_directory, 'screen_log.txt'), 'w')

        sdf_train, pc_train = load_h5(train_file)
        num_batch = len(sdf_train) // cfg.batch_size

        # sdf_test, pc_test = load_h5(test_file)

        # reshape SDF from [64,64,64] -> [64,64,64,1]
        sdf_train = sdf_train.reshape((-1, cfg.cube_len, cfg.cube_len, cfg.cube_len, cfg.cube_channel))
        # sdf_test = sdf_test.reshape((-1, cube_len, cube_len, cube_len,cube_channel))

        printout(flog, 'Loading train file from '+train_file)
        printout(flog, 'SDF_train length '+str(len(sdf_train))+' PLY Train length '+str(len(pc_train)))
        printout(flog, 'num_batch '+ str(num_batch))

        total_steps = 0
        d_acc = 0

        d_avg_real = 0
        d_avg_fake = 0
        d_avg_enc = 0
        d_avg_loss = 0
        g_avg_fake = 0
        g_avg_enc = 0
        g_avg_loss = 0
        e_avg_loss = 0
        rec_avg_loss = 0
        glob_kl_avg_loss = 0
        total_kl_avg_loss = 0
        ds_avg = 0
        dgz_avg = 0
        dgsp_avg = 0
        d_avg_acc = 0

        for epoch in range(cfg.n_epochs):
            # Shuffle data
            idx = np.arange(len(sdf_train))
            np.random.shuffle(idx)
            sdf_train = sdf_train[idx]
            pc_train = pc_train[idx]

            for step in range(num_batch):
                begIdx = step * cfg.batch_size
                endIdx = begIdx + cfg.batch_size
                
                # Sample from uniform distribution
                # z = np.random.uniform(low=-1.0, high=1.0, size=[cfg.batch_size, cfg.z_size]).astype(np.float32)
                # Sample from normal distribution
                z = np.random.normal(size=[cfg.batch_size, cfg.z_size]).astype(np.float32)

                # Run encoder
                # _, _, gen_p, gen_z = sess.run([optimizer_op_e, optimizer_op_g, s_p, s_z], 
                _, _, sample_sdf = sess.run([optimizer_op_e, optimizer_op_g, s_p],
                                                feed_dict = {
                                                        p_vector:pc_train[begIdx:endIdx, ...],
                                                        s_vector:sdf_train[begIdx:endIdx, ...],
                                                        z_vector:z
                                                            })
                if d_acc < cfg.d_thresh_acc:
                    # Run Discriminator
                    sess.run(optimizer_op_d, feed_dict = {
                                                    p_vector:pc_train[begIdx:endIdx, ...],
                                                    s_vector:sdf_train[begIdx:endIdx, ...],
                                                    z_vector:z
                                                        })

                dreal, dFake, dEnc, dLoss, gFake, gEnc, gLoss, eLoss, recLoss,\
                globKLoss, totalKLoss, dS, dGsz, dGsp, d_acc = sess.run([d_real_loss, d_fake_loss, d_encode_loss, d_loss,\
                                                                        g_fake_loss, g_encode_loss, g_loss, e_loss, rec_loss,\
                                                                        glob_kl_loss, total_kl_loss, D_s, D_G_s_z, D_G_s_p, d_accuracy],
                                                                        feed_dict = {
                                                                                p_vector:pc_train[begIdx:endIdx, ...],
                                                                                s_vector:sdf_train[begIdx:endIdx, ...],
                                                                                z_vector:z
                                                                        })
                d_avg_real += dreal
                d_avg_fake += dFake
                d_avg_enc += dEnc
                d_avg_loss += dLoss
                g_avg_fake += gFake
                g_avg_enc += gEnc
                g_avg_loss += gLoss
                e_avg_loss += eLoss
                rec_avg_loss += recLoss
                glob_kl_avg_loss += globKLoss
                total_kl_avg_loss += totalKLoss
                ds_avg += dS
                dgz_avg += dGsz
                dgsp_avg += dGsp
                d_avg_acc += d_acc

                # PrintOut and Summary every 10 steps
                if total_steps % 10 == 0:
                    summary_str = sess.run(summary, feed_dict = {
                                                        p_vector:pc_train[begIdx:endIdx, ...],
                                                        s_vector:sdf_train[begIdx:endIdx, ...],
                                                        z_vector:z
                                                            })
                    summary_writer.add_summary(summary_str, total_steps+1)
                    summary_writer.flush()

                    printout(flog, "##-----------EPOCH %d Step %d Total step: %d-------------###" %(epoch,step+1,total_steps+1))
                    printout(flog, "\td_real_loss: %f" % (d_avg_real/10))
                    printout(flog, "\td_fake_loss: %f" % (d_avg_fake/10))
                    printout(flog, "\td_encoder_loss: %f" % (d_avg_enc/10))
                    printout(flog, "\tTotal D_loss: %f" % (d_avg_loss/10))
                    printout(flog, "\t----------------------------")
                    printout(flog, "\tg_fake_loss: %f" % (g_avg_fake/10))
                    printout(flog, "\tg_encoder_loss: %f" % (g_avg_enc/10))
                    printout(flog, "\tTotal G_loss: %f" % (g_avg_loss/10))
                    printout(flog, "\t----------------------------")
                    printout(flog, "\tTotal E_loss: %f" % (e_avg_loss/10))
                    printout(flog, "\t----------------------------")
                    printout(flog, "\tTotal Rec_loss: %f" % (rec_avg_loss/10))
                    printout(flog, "\t----------------------------")
                    printout(flog, "\tglobal KL loss: %f" % (glob_kl_avg_loss/10))
                    printout(flog, "\tTotal KL loss: %f" % (total_kl_avg_loss/10))
                    printout(flog, "\t----------------------------")
                    printout(flog, "\tD(s): %f" % (ds_avg/10))
                    printout(flog, "\tD(G(s_z)): %f" % (dgz_avg/10))
                    printout(flog, "\tD(G(s_p): %f" % (dgsp_avg/10))
                    printout(flog, "\tD_accuracy: %f\n" % (d_avg_acc/10))
                    d_avg_real = 0
                    d_avg_fake = 0
                    d_avg_enc = 0
                    d_avg_loss = 0
                    g_avg_fake = 0
                    g_avg_enc = 0
                    g_avg_loss = 0
                    e_avg_loss = 0
                    rec_avg_loss = 0
                    glob_kl_avg_loss = 0
                    total_kl_avg_loss = 0
                    ds_avg = 0
                    dgz_avg = 0
                    dgsp_avg = 0
                    d_avg_acc = 0
                
                # output generated objects
                if total_steps % 500 == 0:
                    sam_id = np.random.randint(0, cfg.batch_size, 8)
                    for i in range(8):
                        sample_p = sample_sdf[sam_id[i]].reshape((-1,1))
                        sdf_org = sdf_train[begIdx+sam_id[i]].reshape((-1,1))
                        points = pc_train[begIdx+sam_id[i]]
                        spname = sp_directory+'/gen_sp_'+str(total_steps+1)+'_'+str(i)+'.txt'
                        sname = sdf_directory+'/org_sdf_'+str(total_steps+1)+'_'+str(i)+'.txt'
                        pname = ply_directory+'/ply_'+str(total_steps+1)+'_'+str(i)+'.ply'
                        np.savetxt(spname, sample_p)
                        np.savetxt(sname, sdf_org)
                        write_ply(points, pname)                    

                # store checkpoint
                if total_steps % 2000 == 0:                
                    saver.save(sess, save_path = ckpt_directory + '/gan_model_' + str(total_steps+1) + '.cptk')

                sess.run(tf.assign(global_step, total_steps + 1))
                total_steps = total_steps + 1

            printout(flog, "Global Step %d" %global_step.eval())
        flog.flush()
    flog.close()

if __name__ == '__main__':
    if not os.path.exists(sdf_directory):
        os.makedirs(sdf_directory)
    if not os.path.exists(ply_directory):
        os.makedirs(ply_directory)
    if not os.path.exists(sp_directory):
        os.makedirs(sp_directory)
    if not os.path.exists(ckpt_directory):
        os.makedirs(ckpt_directory)      
    trainGAN()