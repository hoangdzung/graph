import tensorflow as tf
import numpy as np 
import os
import math
import Resnet_spatial
import Resnet_temporal

class Multi_stream_model():
    def __init__(self, queue_data, h5_path, learning_rate=0.1, momentum=0.8):
        self.queue = queue_data
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.lr_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")

        self.spatial_cnn = Resnet_spatial.Resnet50(h5_path=h5_path, n_classes=queue_data.n_classes)
        self.temporal_cnn = Resnet_temporal.Resnet50(h5_path=h5_path, n_classes=queue_data.n_classes)

        self.spatial_cnn.input_matrix, self.temporal_cnn.input_matrix, self.one_hot_label = self.queue.next_batch()
        self.spatial_cnn.is_training = self.temporal_cnn.is_training = self.queue.is_training
        self.spatial_cnn._build_model()
        self.temporal_cnn._build_model()

        self._compute_fusion_score()
        
        self._create_loss()
        self._create_update_center_op()
        self._create_optimizer()
        self._create_evaluater()
        self._create_summarizer()

        self.writer = tf.summary.FileWriter("graph_center")
    
    @staticmethod
    def new_Weight(shape, stddev = 0.01):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    @staticmethod
    def new_Bias(shape, val= 0.0):
        return tf.Variable(tf.constant(val, shape=[shape]))
    
    def _compute_fusion_score(self):
        with tf.device('/gpu:0'):
            self.fusion_score = (self.spatial_cnn.softmax + self.temporal_cnn.softmax)/2
            # http://www.yugangjiang.info/publication/16MM-VideoFusion.pdf
            # self.stacked_score = tf.concat((self.spatial_cnn.score, self.temporal_cnn.score), axis = 1)
            # self.fusion_weight = self.new_Weight((2 * self.n_classes, self.n_classes))
            # self.fusion_bias = self.new_Bias(self.n_classes)
            # self.fusion = tf.matmul(self.stacked_score, self.fusion_weight) + self.fusion_bias
            # self.fusion_score = tf.nn.softmax(self.fusion)

    def _create_loss(self):
        with tf.device('/gpu:0'):
            self.softmax_loss = tf.reduce_mean(-tf.reduce_sum(self.one_hot_label * tf.log(self.fusion_score+1e-15), reduction_indices=[1]))
            self.spatial_centers = tf.get_variable(name = 'center_spatial', shape = [self.queue.n_classes, self.spatial_cnn.n_features],
                                        initializer = tf.constant_initializer(0.0), trainable = False)
            self.temporal_centers= tf.get_variable(name = 'center_temporal', shape = [self.queue.n_classes, self.temporal_cnn.n_features],
                                        initializer = tf.constant_initializer(0.0), trainable = False)
            self.label = tf.argmax(self.one_hot_label, axis = 1)
            
            self.spatial_center_matrix = tf.gather(self.spatial_centers, self.label)
            self.temporal_center_matrix = tf.gather(self.temporal_centers, self.label)

            self.spatial_diff = self.spatial_center_matrix - self.spatial_cnn.embedding
            self.temporal_diff = self.temporal_center_matrix - self.temporal_cnn.embedding
            self.center_loss = tf.nn.l2_loss(self.spatial_diff) + tf.nn.l2_loss(self.temporal_diff) 
            
            self.loss = self.softmax_loss + 0.01 * self.center_loss
    
    def _create_update_center_op(self):
        unique_label, unique_idx, unique_count = tf.unique_with_counts(self.label)
        self.appear_times = tf.gather(unique_count, unique_idx) + 1
        self.appear_times = tf.cast(self.appear_times, tf.float32)
        self.appear_times = tf.reshape(self.appear_times, [-1, 1])
        self.spatial_update = tf.div(self.spatial_diff, self.appear_times)
        self.temporal_update = tf.div(self.temporal_diff, self.appear_times)
        self.spatial_centers_update_op = tf.scatter_sub(self.spatial_centers, self.label, self.spatial_update)
        self.temporal_centers_update_op = tf.scatter_sub(self.temporal_centers, self.label, self.temporal_update)


    def _create_optimizer(self):
        with tf.device('/gpu:0'):
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=self.momentum, use_nesterov=True).minimize(self.loss,
                                                                                                global_step = self.global_step)
                                                                                        
    def _create_evaluater(self):
        with tf.device('/gpu:0'):
            correct = tf.equal(tf.argmax(self.fusion_score, 1), tf.argmax(self.one_hot_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    def _create_summarizer(self):
        self.train_acc_placeholder = tf.placeholder(dtype = tf.float32, shape =())
        tf.summary.scalar("train_accuracy", self.train_acc_placeholder)

        self.test_acc_placeholder = tf.placeholder(dtype = tf.float32, shape =())
        tf.summary.scalar("test_accuracy", self.test_acc_placeholder)
        
        self.train_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_loss", self.train_loss_placeholder)

        self.test_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_loss", self.test_loss_placeholder)

        self.merged_summary = tf.summary.merge_all()
