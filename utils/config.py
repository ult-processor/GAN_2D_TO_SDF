import tensorflow as tf

flags = tf.app.flags


########################
#	Hyper parameters.  #
########################
# number of steps for each class should be ~43k steps
# learning rate decay for 2 times
# Chair: n_batch = 43, n_epoch = 1000
# Table: n_batch = 18, n_epoch = 2400
# Sofa:  n_batch = 34, n_epoch = 1265
# Car:   n_batch = 9,  n_epoch = 4800
########################

# For training
flags.DEFINE_boolean('is_training', True, 'Train or Test')
flags.DEFINE_integer('n_epochs', 1000, 'num of epochs')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('z_size', 128, 'Latent vector size')

flags.DEFINE_float('init_d_lr', 1e-4, 'Discriminator initial learning rate')
flags.DEFINE_float('init_g_lr', 2e-3, 'Generator initial learning rate')
flags.DEFINE_float('init_e_lr', 1e-4, 'Encoder initial learning rate')

flags.DEFINE_integer('DECAY_STEP', 10000, 'Decay step')
flags.DEFINE_float('D_DECAY_RATE', 0.5, 'Discriminator decay rate')
flags.DEFINE_float('G_DECAY_RATE', 0.5, 'Generator decay rate')
flags.DEFINE_float('E_DECAY_RATE', 0.5, 'Encoder decay rate')

flags.DEFINE_float('d_thresh_acc', 0.8, 'Dis training accuracy threshold')

# ops paramenter
flags.DEFINE_float('momentum', 0.9, 'Batch normal. momentum')
flags.DEFINE_float('leak_value', 0.2, 'Leak relu parameters')
flags.DEFINE_float('beta', 0.5, 'optimizer paramenter')

# Input data parameters
flags.DEFINE_integer('num_points', 1024, 'num of points in point cloud')
flags.DEFINE_integer('cube_len', 64, 'Cube lenght')
flags.DEFINE_integer('cube_channel', 1, 'cube channel')

# Loss parameters
flags.DEFINE_float('alpha1', 5.0, 'KL loss scale')
flags.DEFINE_float('alpha2', 10.0, 'Reconstruction loss scale')

cfg = tf.app.flags.FLAGS
