import tensorflow as tf
import os
layers = tf.contrib.layers

def batch_norm_layer(inputs, phase_train, scope=None):
    if phase_train:
        return layers.batch_norm(inputs, is_training=True, scale=True,
            updates_collections=None, scope=scope)
    else:
        return layers.batch_norm(inputs, is_training=False, scale=True,
            updates_collections=None, scope=scope, reuse = True)

def conv(inputs, name, train, filter_num, kernel_size, strides):
    conv_out = layers.convolution2d(inputs, filter_num, kernel_size=kernel_size, stride=strides, 
    padding='VALID', activation_fn=None, data_format='NHWC', scope=name+"_conv")
    conv_out = batch_norm_layer(conv_out, train, scope=name+"_bn")
    conv_out = tf.nn.relu(conv_out)
    return conv_out

#noise_shape_mode 
# 0 -> 
def conv_with_dropout(inputs, name, train, filter_num, kernel_size, strides, dropout_prob, noise_shape_mode):
    conv_out = conv(inputs, name, train, filter_num, kernel_size, strides)
    conv_shape = conv_out.get_shape().as_list()
    noise_shape = []
    if noise_shape_mode == 0:
        noise_shape = [conv_shape[0], 1, 1, conv_shape[3]]
    conv_out = layers.dropout(conv_out, dropout_prob, is_training=train, 
    noise_shape=noise_shape, scope=name+'_dropout')
    return conv_out