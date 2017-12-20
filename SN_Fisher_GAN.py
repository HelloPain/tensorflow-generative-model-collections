#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from ops import *
from libs.ops import *
from utils import *
from glob import glob
import scipy.misc as misc

class SN_FisherGAN(object):
    model_name = "SN_FisherGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1
            self.alpha = tf.get_variable("fisher_gan_alpha", [], initializer=tf.zeros_initializer)
            self.rho = 1e-6

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        elif dataset_name == 'CelebA':
                # parameters
            self.reshape_input_height = 64
            self.reshape_input_width = 64
            self.output_height = 64
            self.output_width = 64
            self.lambd = 0.25
            self.z_dim = z_dim  # dimension of noise-vector
            self.c_dim = 3
            self.alpha = tf.get_variable("fisher_gan_alpha", [], initializer=tf.zeros_initializer)
            self.rho = 1e-6
            self.beta1 = 0.5

            #magic num
            self.magic_num = 16
            # BEGAN Parameter
            self.gamma = 0.5
            self.lamda = 0.001

            # train
            
            self.learning_rate = tf.Variable(0.0001, name='lr')
            self.lr_update = tf.assign(self.learning_rate, tf.maximum(self.learning_rate * 0.5, 0.00002), name='lr_update')
            
            # test
            self.sample_num = 64  # number of generated images to be saved


            self.data_X = glob(os.path.join("./data/CelebA/splits/train/*.jpg"))
            h, w, _ = misc.imread(self.data_X[0]).shape
            self.input_height = h
            self.input_width = w
            #self.data_y = np.concatenate([y_train, y_test])

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, update_collection=None, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        hidden_activation = lrelu
        with tf.variable_scope("discriminator", reuse=reuse) as vs:
            c0_0 = hidden_activation(conv2d(   x,  64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_0'))
            c0_1 = hidden_activation(conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c0_1'))
            c1_0 = hidden_activation(conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_0'))
            c1_1 = hidden_activation(conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1_1'))
            c2_0 = hidden_activation(conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_0'))
            c2_1 = hidden_activation(conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2_1'))
            c3_0 = hidden_activation(conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3_0'))
            c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
            out_logit = linear(c3_0, self.c_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
            out = tf.nn.sigmoid(out_logit)
        variables = tf.contrib.framework.get_variables(vs)
        return out, out_logit, c2_0, variables

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        hidden_activation = tf.nn.relu
        output_activation = tf.nn.tanh
        with tf.variable_scope("generator", reuse=reuse) as vs:
            l0  = hidden_activation(batch_norm(linear(z, 4 * 4 * 512, name='l0', stddev=0.02), name='bn0', is_training=is_training))
            l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
            dc1 = hidden_activation(batch_norm(deconv2d( l0, [self.batch_size,  8,  8, 256], name='dc1', stddev=0.02), name='bn1', is_training=is_training))
            dc2 = hidden_activation(batch_norm(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02), name='bn2', is_training=is_training))
            dc3 = hidden_activation(batch_norm(deconv2d(dc2, [self.batch_size, 32, 32,  64], name='dc3', stddev=0.02), name='bn3', is_training=is_training))
            dc4 = hidden_activation(batch_norm(deconv2d(dc3, [self.batch_size, 64, 64,  32], name='dc4', stddev=0.02), name='bn4', is_training=is_training))
            dc5 = deconv2d(dc4, [self.batch_size, 64, 64, 3], 3, 3, 1, 1, name='dc5', stddev=0.02)
        variables = tf.contrib.framework.get_variables(vs)
        return dc5, variables

    def build_model(self):
        # some parameters
        image_dims = [self.reshape_input_height, self.reshape_input_width, self.c_dim]
        #image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _, d_vars = self.discriminator(self.inputs, is_training=True, update_collection=None, reuse=False)

        # output of D for fake images
        G, g_vars = self.generator(self.z, is_training=True, reuse=False)
        D_fake, D_fake_logits, _, _ = self.discriminator(G, is_training=True, update_collection="NO_OPS", reuse=True)

        # get loss for discriminator
        E_Q_f = tf.reduce_mean(D_fake_logits)
        E_P_f = tf.reduce_mean(D_real_logits)
        E_Q_f2 = tf.reduce_mean(D_fake_logits**2)
        E_P_f2 = tf.reduce_mean(D_real_logits**2)

        constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

        self.d_loss = -1.0 * (E_P_f - E_Q_f + self.alpha * constraint - self.rho/2 * constraint**2)

        # get loss for generator
        self.g_loss = -tf.reduce_mean(D_fake_logits)

        """ Training """
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)
            self.alpha_optim = tf.train.GradientDescentOptimizer(self.rho) \
            		  .minimize(-self.d_loss, var_list=[self.alpha])

        """" Testing """
        # for test
        self.fake_images, _ = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                if self.dataset_name == 'CelebA':
                    batch_files = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                    files = [get_celeba_image(batch_file,
                                              input_height=self.input_height,
                                              input_width=self.input_width,
                                              resize_height=self.output_height,
                                              resize_width=self.output_width,
                                              crop=False,
                                              grayscale=False) for batch_file in batch_files]
                    batch_images = np.array(files).astype(np.float32)
                    #print(batch_images.shape)
                else:
                    batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                _, _, summary_str, d_loss = self.sess.run([self.d_optim, self.alpha_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    #print(samples.shape)
                    #print(manifold_h, manifold_w)
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
