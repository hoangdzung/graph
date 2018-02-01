import tensorflow as tf
import numpy as np 
import os
import math
class CNN_model():
    def __init__(self, name, n_classes, batch_size = 32, gpu=None):
        # self.input_shape = input_shape
        self.name = name
        self.gpu = gpu
        self.input_matrix = None
        # self.input = tf.placeholder(dtype = tf.float32,
        #                             shape = [None,
        #                                     self.input_shape[0],
        #                                     self.input_shape[1],
        #                                     self.input_shape[2]],
        #                             name = 'input')
        self.n_classes = n_classes
        self.batch_size = batch_size
        # self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
        # self._build_model()
        
    @staticmethod
    def new_Weight(shape, stddev = 0.001):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

    @staticmethod
    def new_Bias(shape, value = 0.0):
        return tf.Variable(tf.constant(value, shape=[shape]))

    def _conv2d(self, input, filter_size, n_filters, stride):
        n_input_channels = int(input.shape[-1])
        shape = [filter_size, filter_size, n_input_channels, n_filters]
        
        weights = self.new_Weight(shape, math.sqrt(3.0/(n_input_channels+n_filters)))
        biases = self.new_Bias(n_filters)

        layer = tf.nn.conv2d(input = input,
                            filter = weights,
                            strides = [1, stride, stride, 1],
                            padding = 'SAME')
        layer += biases
        layer = tf.nn.relu(layer)
        #layer = tf.Print(layer, [layer], message="conv2d_"+self.name)
        return layer
    
    def _batch_norm(self, input):
        batch_mean, batch_var = tf.nn.moments(input ,[0,1,2])
        depth = int(input.shape[-1])
        scale = self.new_Bias(depth, 1.0)
        beta = self.new_Bias(depth)

        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(tf.cast(self.is_training, tf.bool),
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))

        layer = tf.nn.batch_normalization(input, mean, var , beta, scale, 1e-12)
        #layer = tf.Print(layer, [layer], message="batch_norm_"+self.name)
        return layer

    def _maxpooling2d_layer(self, input,size, strides):
        layer = tf.nn.max_pool(value = input,
                            ksize=[1, size, size, 1],
                            strides=[1, strides, strides, 1],
                            padding='VALID')
        #layer = tf.Print(layer, [layer], message="max_pool_"+self.name)                    
        return layer

    def _flatten_layer(self, input):
        h = int(input.shape[1])
        w = int(input.shape[2])
        d = int(input.shape[3])
        return tf.reshape(input, [-1, h*w*d])
    
    def _fully_connected_layer(self, input, n_outputs, keep_rate = None, activation = 'relu'):
        n_inputs = int(input.shape[-1])
        weights = self.new_Weight([n_inputs, n_outputs], math.sqrt(3.0/(n_inputs+n_outputs)))
        biases = self.new_Bias(n_outputs)

        layer = tf.matmul(input, weights) + biases

        if activation == 'softmax':
            layer = tf.nn.softmax(layer)
        elif activation == 'relu':
            layer = tf.nn.relu(layer)
        if keep_rate != None:
            layer = tf.nn.dropout(layer, keep_rate)
        #layer = tf.Print(layer, [layer], message="fc_"+activation+"_"+ self.name)
        return layer

    def _build_model(self):
        with tf.device('/gpu:%d'%self.gpu):
            self.conv1a = self._conv2d(self.input_matrix, 7, 96, 2)
            self.conv1b = self._batch_norm(self.conv1a)
            self.conv1c = self._maxpooling2d_layer(self.conv1b, 2, 2)
            
            self.conv2a = self._conv2d(self.conv1c, 5, 256, 2)
            self.conv2b = self._batch_norm(self.conv2a)
            self.conv2c = self._maxpooling2d_layer(self.conv2b, 2, 2)

            self.conv3 = self._conv2d(self.conv2c, 3, 512, 1)
            self.conv4 = self._conv2d(self.conv3, 3, 512, 1)

            self.conv5a = self._conv2d(self.conv4, 3, 512, 1)
            self.conv5b = self._maxpooling2d_layer(self.conv5a, 2, 2)
            self.conv5c = self._flatten_layer(self.conv5b)

            self.full6 = self._fully_connected_layer(self.conv5c, 4096, 0.8, 'relu')
            self.full7 = self._fully_connected_layer(self.full6, 2048, 0.8, 'relu')
            self.score = self._fully_connected_layer(self.full7, self.n_classes)
            self.l2_norm_score = tf.nn.l2_normalize(self.score, 1)
            self.softmax_score = tf.nn.softmax(self.score)

class Multi_stream_model():
    def __init__(self, queue_data, learning_rate=0.1, momentum=0.8):
        self.queue = queue_data
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.lr_placeholder = tf.placeholder(dtype = tf.float32, shape = ())
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")

        self.spatial_cnn = CNN_model(n_classes=queue_data.n_classes, name='spatial',batch_size=queue_data.batch_size, gpu=0)
        self.temporal_cnn = CNN_model(n_classes=queue_data.n_classes, name='temporal',batch_size = queue_data.batch_size, gpu=1)

        self.spatial_cnn.input_matrix, self.temporal_cnn.input_matrix, self.one_hot_label = self.queue.next_batch()
        self.spatial_cnn.is_training = self.temporal_cnn.is_training = self.queue.is_training
        self.spatial_cnn._build_model()
        self.temporal_cnn._build_model()

        self._compute_fusion_score()
        
        self._create_loss()
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
            self.fusion_score = (self.spatial_cnn.l2_norm_score + self.temporal_cnn.l2_norm_score)/2
            # http://www.yugangjiang.info/publication/16MM-VideoFusion.pdf
            # self.stacked_score = tf.concat((self.spatial_cnn.score, self.temporal_cnn.score), axis = 1)
            # self.fusion_weight = self.new_Weight((2 * self.n_classes, self.n_classes))
            # self.fusion_bias = self.new_Bias(self.n_classes)
            # self.fusion = tf.matmul(self.stacked_score, self.fusion_weight) + self.fusion_bias
            # self.fusion_score = tf.nn.softmax(self.fusion)

    def _create_loss(self):
        with tf.device('/gpu:0'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.one_hot_label * tf.log(self.fusion_score+1e-15), reduction_indices=[1]))

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

