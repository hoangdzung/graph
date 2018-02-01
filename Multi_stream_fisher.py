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

        self.writer = tf.summary.FileWriter("graph")
    
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
            
            self.spatial_centers = tf.get_variable(name = 'centers_spatial', shape = [self.queue.n_classes, self.spatial_cnn.n_features],
                                        initializer = tf.constant_initializer(0.0), trainable = False)
            self.temporal_centers= tf.get_variable(name = 'centers_temporal', shape = [self.queue.n_classes, self.temporal_cnn.n_features],
                                        initializer = tf.constant_initializer(0.0), trainable = False)
            self.label = tf.argmax(self.one_hot_label, axis = 1)
            
            self.spatial_center_matrix = tf.gather(self.spatial_centers, self.label)
            self.temporal_center_matrix = tf.gather(self.temporal_centers, self.label)
            
            self.spatials_diff = self.spatial_center_matrix - self.spatial_cnn.embedding
            self.spatials_loss = tf.nn.l2_loss(self.spatials_diff)
            
            self.temporals_diff = self.temporal_center_matrix - self.temporal_cnn.embedding
            self.temporals_loss = tf.nn.l2_loss(self.temporals_diff)
            
            self.spatial_center = tf.get_variable(name="center_spatial", shape=(self.spatial_cnn.n_features,),
                                     initializer = tf.constant_initializer(0.0), trainable = False)
            self.temporal_center = tf.get_variable(name="center_temporal", shape=(self.temporal_cnn.n_features,),
                                     initializer = tf.constant_initializer(0.0), trainable = False)
            
            self.spatial_diff = self.spatial_center_matrix- self.spatial_center
            self.spatial_loss= tf.nn.l2_loss(self.spatial_diff)
            
            self.temporal_diff = self.temporal_center_matrix- self.temporal_center
            self.temporal_loss= tf.nn.l2_loss(self.temporal_diff)
            
            self.spatial_center_loss = tf.maximum(self.spatials_loss - 0.001*self.spatial_loss,0)
            self.temporal_center_loss = tf.maximum(self.temporals_loss - 0.001*self.temporal_loss,0) 

            self.loss = self.softmax_loss + 0.1 * (self.spatial_center_loss + self.temporal_center_loss)
    def _create_update_center_op(self):
        unique_label, unique_idx, unique_count = tf.unique_with_counts(self.label)
        self.appear_times = tf.gather(unique_count, unique_idx) + 1
        self.appear_times = tf.cast(self.appear_times, tf.float32)
        self.appear_times = tf.reshape(self.appear_times, [-1, 1])
        self.spatials_update = tf.div(self.spatials_diff - 0.001*self.spatial_diff, self.appear_times)
        self.temporals_update = tf.div(self.temporals_diff - 0.001*self.temporal_diff, self.appear_times)
        self.spatial_centers_update_op = tf.scatter_sub(self.spatial_centers, self.label, self.spatials_update)
        self.temporal_centers_update_op = tf.scatter_sub(self.temporal_centers, self.label, self.temporals_update)

        self.spatial_update = tf.reduce_sum(tf.div(self.spatial_cnn.embedding- self.spatial_center-0.001*self.spatial_diff, self.appear_times), axis=0)
        self.temporal_update = tf.reduce_sum(tf.div(self.temporal_cnn.embedding- self.temporal_center-0.001*self.temporal_diff, self.appear_times), axis=0)

        self.spatial_center_update_op=tf.scatter_add(self.spatial_center, np.arange(self.spatial_cnn.n_features), 0.001*self.spatial_update) 
        self.temporal_center_update_op=tf.scatter_add(self.temporal_center, np.arange(self.temporal_cnn.n_features), 0.001*self.temporal_update) 
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
        
        # Train loss
        self.train_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_loss", self.train_loss_placeholder)
        
        self.train_softmax_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_softmax_loss", self.train_softmax_loss_placeholder)
        
        self.train_spatials_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_spatials_loss", self.train_spatials_loss_placeholder)
        
        self.train_spatial_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_spatial_loss", self.train_spatial_loss_placeholder)
        
        self.train_temporals_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_temporals_loss", self.train_temporals_loss_placeholder)
        
        self.train_temporal_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("train_temporal_loss", self.train_temporal_loss_placeholder)
        
        # Test loss
        self.test_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_loss", self.test_loss_placeholder)
        
        self.test_softmax_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_softmax_loss", self.test_softmax_loss_placeholder)
        
        self.test_spatials_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_spatials_loss", self.test_spatials_loss_placeholder)
        
        self.test_spatial_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_spatial_loss", self.test_spatial_loss_placeholder)
        
        self.test_temporals_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_temporals_loss", self.test_temporals_loss_placeholder)
        
        self.test_temporal_loss_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        tf.summary.scalar("test_temporal_loss", self.test_temporal_loss_placeholder)

        self.merged_summary = tf.summary.merge_all()
