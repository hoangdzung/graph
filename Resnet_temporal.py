from __future__ import print_function
import tensorflow as tf
import numpy as np 
# from dataset import Dataset
import math
import os
import cv2 
import sys
import h5py

class Resnet50():
    def __init__(self, h5_path, n_classes, n_features = 512, name = None):
        """
            name: string, name of model using for saving graph
            data_h5: string, path of pretrained weights file
            n_classes: int, number of classes
            n_features: int, size of embedded features
        """

        self.name = name
        self.data_h5 = h5py.File(h5_path, 'r')
        self.n_classes = n_classes
        self.n_features = n_features
        self.is_training = tf.placeholder(tf.int32, [])
    def build(self):
        raise NotImplementedError
        
    def _create_placeholder(self):
        raise NotImplementedError  
    
    @staticmethod
    def new_Weight(shape, stddev=0.01):
        return tf.Variable(tf.truncated_normal(shape = shape, stddev=stddev))

    @staticmethod
    def new_Bias(shape, value = 0.0):
        return tf.Variable(tf.constant(value, shape=shape))
    @staticmethod
    def conv_layer(input_layer, data, layer_name, block, strides = [1,1,1,1], padding = 'VALID', use_relu = False):
        with tf.variable_scope(layer_name):
            if block:
                W = tf.constant(data[layer_name][layer_name+'_W_1:0'])
                b = data[layer_name][layer_name+'_b_1:0']
                b = tf.constant(np.reshape(b,(b.shape[0])))
            else:
                W = tf.Variable(data[layer_name][layer_name+'_W_1:0'])
                b = data[layer_name][layer_name+'_b_1:0']
                b = tf.Variable(np.reshape(b,(b.shape[0])))
            x = tf.nn.conv2d(input_layer, filter=W, strides=strides, padding=padding, name=layer_name)
            x = tf.nn.bias_add(x, b)
            return x

    @staticmethod
    def batch_norm_layer_(input_layer, data, layer_name, block):
        with tf.variable_scope(layer_name):
            if block:
                beta = tf.constant( data[layer_name][layer_name+'_beta_1:0'] )
                gamma = tf.constant( data[layer_name][layer_name+'_gamma_1:0'] )
                mean = tf.constant( data[layer_name][layer_name+'_running_mean_1:0'] )
                std = tf.constant( data[layer_name][layer_name+'_running_std_1:0'] )
            else:
                beta = tf.Variable( data[layer_name][layer_name+'_beta_1:0'] )
                gamma = tf.Variable( data[layer_name][layer_name+'_gamma_1:0'] )
                mean, std = tf.nn.moments(input_layer ,[0, 1, 2])
            return tf.nn.batch_normalization(
                input_layer, mean=mean, variance=std, 
                offset=beta, scale=gamma, 
                variance_epsilon=1e-12, name='batch-norm')
    
    def batch_norm_layer(self, input_layer, data, layer_name, block):
        with tf.variable_scope(layer_name):
            if block:
                beta = tf.constant( data[layer_name][layer_name+'_beta_1:0'] )
                gamma = tf.constant( data[layer_name][layer_name+'_gamma_1:0'] )
                mean = tf.constant( data[layer_name][layer_name+'_running_mean_1:0'] )
                std = tf.constant( data[layer_name][layer_name+'_running_std_1:0'] )
            else:
                beta = tf.Variable( data[layer_name][layer_name+'_beta_1:0'] )
                gamma = tf.Variable( data[layer_name][layer_name+'_gamma_1:0'] )
                batch_mean, batch_std = tf.nn.moments(input_layer ,[0, 1, 2])
               
                ema = tf.train.ExponentialMovingAverage(decay=0.5)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_std])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_std)

                mean, std = tf.cond(tf.cast(self.is_training, tf.bool),
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_std)))

            return tf.nn.batch_normalization(
                input_layer, mean=mean, variance=std, 
                offset=beta, scale=gamma, 
                variance_epsilon=1e-12, name='batch-norm')

    def identity_block(self, input_layer, stage, data, block = False):
        
        with tf.variable_scope('identity_block'):
            
            x = self.conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch2a', block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2a', block = block)
            x = tf.nn.relu(x)
            
            x = self.conv_layer(x, data=data, layer_name='res'+stage+'_branch2b', padding='SAME', block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2b', block = block)
            x = tf.nn.relu(x)
            
            x = self.conv_layer(x, data=data, layer_name='res'+stage+'_branch2c', block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2c', block = block)
            
            x = tf.add(x, input_layer)
            x = tf.nn.relu(x)
            
        return x

    def conv_block(self, input_layer, stage, data, block = False, strides=[1, 2, 2, 1]):
    
        with tf.variable_scope('conv_block'):
            
            x = self.conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch2a', strides=strides, block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2a', block = block)
            x = tf.nn.relu(x)
            
            x = self.conv_layer(x, data=data, layer_name='res'+stage+'_branch2b', padding='SAME', block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2b', block = block)
            x = tf.nn.relu(x)
            
            x = self.conv_layer(x, data=data, layer_name='res'+stage+'_branch2c', block = block)
            x = self.batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2c', block = block)
            
            shortcut = self.conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch1', strides=strides, block = block)
            shortcut = self.batch_norm_layer(shortcut, data=data, layer_name='bn'+stage+'_branch1', block = block)
            
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x)
            
        return x

    def _flatten_layer(self, input):
        h = int(input.shape[1])
        w = int(input.shape[2])
        d = int(input.shape[3])
        return tf.reshape(input, [-1, h*w*d])

    def _fully_connected_layer(self, input, n_outputs, activation = None, drop_rate = None):
        n_inputs = int(input.shape[-1])
        weights = self.new_Weight([n_inputs, n_outputs], stddev = math.sqrt(3.0/(n_inputs+n_outputs)))
        biases = self.new_Bias([n_outputs])

        layer = tf.matmul(input, weights) + biases

        if activation == 'softmax':
            layer = tf.nn.softmax(layer)
        elif activation == 'relu':
            layer = tf.nn.relu(layer)
        if drop_rate != None:
            layer = tf.nn.dropout(layer, drop_rate)
        return layer

    def _build_model(self):
        data_h5 = self.data_h5
        with tf.device("/gpu:0"):
            with tf.variable_scope('stage1'):
                with tf.variable_scope('conv1'):
                    W = data_h5['conv1']['conv1_W_1:0']
                    W_expand = [W for i in range(7)]
                    W = np.concatenate(W_expand, axis=2)
                    W = W[:,:,:20,:]
                    W = tf.Variable(W)
                    b = data_h5['conv1']['conv1_b_1:0']
                    b = tf.Variable(np.reshape(b,(b.shape[0])))
                    res = tf.nn.conv2d(self.input_matrix, filter=W, strides=[1, 2, 2, 1], padding='VALID', name='conv')
                    res = tf.nn.bias_add(res, b)

                res = self.batch_norm_layer(res, data_h5, 'bn_conv1',False)
                res = tf.nn.relu(res)
                res = tf.nn.max_pool(res, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_conv1')

            with tf.variable_scope('stage2'):
                res = self.conv_block(input_layer=res, stage='2a', data=data_h5, strides=[1, 1, 1, 1])
                res = self.identity_block(input_layer=res, stage='2b', data=data_h5)
                res = self.identity_block(input_layer=res, stage='2c', data=data_h5)

            with tf.variable_scope('stage3'):
                res = self.conv_block(input_layer=res, stage='3a', data=data_h5)
                res = self.identity_block(input_layer=res, stage='3b', data=data_h5)
                res = self.identity_block(input_layer=res, stage='3c', data=data_h5)
                res = self.identity_block(input_layer=res, stage='3d', data=data_h5)

            with tf.variable_scope('stage4'):
                res = self.conv_block(input_layer=res, stage='4a', data=data_h5)
                res = self.identity_block(input_layer=res, stage='4b', data=data_h5)
                res = self.identity_block(input_layer=res, stage='4c', data=data_h5)
                res = self.identity_block(input_layer=res, stage='4d', data=data_h5)
                res = self.identity_block(input_layer=res, stage='4e', data=data_h5)
                res = self.identity_block(input_layer=res, stage='4f', data=data_h5)

            with tf.variable_scope('stage5'):
                res = self.conv_block(input_layer=res, stage='5a', data=data_h5)
                res = self.identity_block(input_layer=res, stage='5b', data=data_h5)
                res = self.identity_block(input_layer=res, stage='5c', data=data_h5)

            with tf.variable_scope('stage-final'):
                res = tf.nn.avg_pool(res, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool_conv1')
                res = tf.reshape(res, (-1, res.get_shape()[3].value))
                
                self.full1 = self._fully_connected_layer(res, 4096, 'relu', 0.5)
                self.full2 = self._fully_connected_layer(self.full1, self.n_features, 'relu', 0.5)
                self.embedding = tf.nn.l2_normalize(self.full2, 1)
                self.softmax = self._fully_connected_layer(self.full2, self.n_classes, 'softmax')

    def _create_loss(self):
        raise NotImplementedError
        
    def _create_optimizer(self):
        raise NotImplementedError

    def _create_evaluater(self):
        raise NotImplementedError         
