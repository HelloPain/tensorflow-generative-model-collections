#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import scipy.misc as misc
import tensorflow as tf
import numpy as np

from glob import glob
from ops import *
from utils import *

class BEGAN(object):
    model_name = "BEGAN"     # name for checkpoint

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

            # BEGAN Parameter
            self.gamma = 0.5
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0001
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

    def discriminator(self, x_, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as vs:
            # Encoder
            hidden_num = 128
            if self.dataset_name == 'CelebA':
                repeat_num = 4
            x = slim.conv2d(x_, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            prev_channel_num = hidden_num
            
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < repeat_num-1:
                    channel_num2 = channel_num+hidden_num
                else:
                    channel_num2 = channel_num
                x = slim.conv2d(x, channel_num2, 3, 1, activation_fn=tf.nn.elu)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num2, 3, 2, activation_fn=tf.nn.elu)

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            code = x = slim.fully_connected(x, self.z_dim, activation_fn=None)

            # Decoder
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = reshape(x, 8, 8, hidden_num)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, hidden_num*2, 1, 1, activation_fn=tf.nn.elu)
                    x = upscale(x, 2)
                    
            out = slim.conv2d(x, self.c_dim, 3, 1, activation_fn=None)
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x_)) / self.batch_size

            variables = tf.contrib.framework.get_variables(vs)
            return out, recon_error, code, variables

    def generator(self, z, is_training=True, reuse=False):
        hidden_num = 128
        if self.dataset_name == 'CelebA':
            repeat_num = 4
        with tf.variable_scope("generator", reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = tf.reshape(x, [-1, 8, 8, hidden_num])

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, hidden_num*2, 1, 1, activation_fn=tf.nn.elu)
                    x = upscale(x, 2)

            out = slim.conv2d(x, 3, 3, 1, activation_fn=None)
            variables = tf.contrib.framework.get_variables(vs)
            return out, variables

    def build_model(self):
        # some parameters
        image_dims = [self.reshape_input_height, self.reshape_input_width, self.c_dim]
        bs = self.batch_size

        """ BEGAN variable """
        self.k = tf.Variable(0., trainable=False)

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real_img, D_real_err, D_real_code, self.d_vars = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G, self.g_vars = self.generator(self.z, is_training=True, reuse=False)
        D_fake_img, D_fake_err, D_fake_code, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        self.d_loss = D_real_err - self.k*D_fake_err

        # get loss for generator
        self.g_loss = D_fake_err

        # convergence metric
        self.M = D_real_err + tf.abs(self.gamma*D_real_err - D_fake_err)

        # operation for updating k
        self.update_k = self.k.assign(
            tf.clip_by_value(self.k + self.lamda*(self.gamma*D_real_err - D_fake_err), 0, 1))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate) \
                      .minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate) \
                      .minimize(self.g_loss, var_list=self.g_vars)

        """" Testing """
        # for test
        self.fake_images, _ = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_error_real", D_real_err)
        d_loss_fake_sum = tf.summary.scalar("d_error_fake", D_fake_err)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        M_sum = tf.summary.scalar("M", self.M)
        k_sum = tf.summary.scalar("k", self.k)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.p_sum = tf.summary.merge([M_sum, k_sum])

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
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update k
                _, summary_str, M_value, k_value = self.sess.run([self.update_k, self.p_sum, self.M, self.k], feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, M: %.8f, k: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, M_value, k_value))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
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
