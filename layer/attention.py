import tensorflow as tf
import numpy as np
import time
import math
import os
import sys
from sklearn import metrics
from tensorflow.python.layers.core import Dropout, Dense
layers = tf.contrib.layers

def attention_fusion_1(inputs):
	"""inputs, shape: [batch, input_mode, feature_dim]"""
	global attention_weight
	d = inputs.shape.as_list()[2]
	w = tf.Variable(tf.random_normal([d], stddev=0.1))
	b = tf.Variable(tf.random_normal([]))
	print "variable_name_b " + b.name
	#activation = tf.tanh(tf.matmul(inputs, w) + b)  # b *  i
	activation = tf.tanh(tf.tensordot(inputs, w, axes=1) + b)  # b *  i
	alphas = tf.nn.softmax(activation)  # b * i
	# attention_weight = alphas
	output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
	print("alphas ", str(output.shape))
	return output


def attention_fusion_2(inputs):
	"""inputs, shape: [batch, time_step, input_mode, feature_dim]"""
	global attention_weight
	d = inputs.shape.as_list()[3]
	w = tf.Variable(tf.random_normal([d], stddev=0.1))
	b = tf.Variable(tf.random_normal([]))
	print "variable_name_w " + w.name
	#activation = tf.tanh(tf.matmul(inputs, w) + b)  # b * t *  i
	activation = tf.tanh(tf.tensordot(inputs, w, axes=1) + b)  # b * t *  i
	alphas = tf.nn.softmax(activation)  # b * t * i
	# attention_weight = alphas
	attention_weight.append(alphas)
	print("alphas ", str(alphas.shape))
	output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 2)
	return output


def attention(inputs, attention_size, time_major=False, return_alphas=False):
	if isinstance(inputs, tuple):
		# In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
		inputs = tf.concat(inputs, 2)

	if time_major:
		# (T,B,D) => (B,T,D)
		inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

	hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

	# Trainable parameters
	w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
	b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	print "variable_name", b_omega.name

	with tf.name_scope('v'):
		# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
		#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
		v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

	# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
	vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
	alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
	# attention_weight = alphas

	# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
	output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

	if not return_alphas:
		return output
	else:
		return output, alphas




attention_weight = []
def self_attention(inputs, name):
	"""
	:param inputs_a: audio input (B, T, dim)
	:param inputs_v: video input (B, T, dim)
	:param inputs_t: text input (B, T, dim)
	:param name: scope name
	:return:
	"""
	# inputs = (B, T, 3, dim)
	# inputs = (B, 3, T, dim) old
	t = inputs.get_shape()[1].value
	share_param = True
	hidden_size = inputs.shape[-1].value  # D value - hidden size of the RNN layer
	if share_param:
		scope_name = 'self_attn'
	else:
		scope_name = 'self_attn' + name
	# print(scope_name)
	#inputs = tf.transpose(inputs, [2, 0, 1, 3])#TB3D
	inputs = tf.transpose(inputs, [1, 0, 2, 3])
	global attention_weight
	attention_weight = []
	with tf.variable_scope(scope_name):
		outputs = []
		for x in range(t):
			t_x = inputs[x, :, :, :]
			# t_x => B, 3, dim
			den = True
			if den:
				x_proj = Dense(hidden_size)(t_x)
				x_proj = tf.nn.tanh(x_proj)
			else:
				x_proj = t_x
			#x_proj B,3,D
			u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.1, seed=1234))
			#u_w D,1
			x = tf.tensordot(x_proj, u_w, axes=1)
			#x B,3,1
			x = tf.reshape(x, [x.shape[0], x.shape[1]])
			# alpha B, 2
			alphas = tf.nn.softmax(x)
			attention_weight.append(alphas)
			output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), tf.reshape(alphas, [x.shape[0], x.shape[1], 1]))
			output = tf.squeeze(output, -1)
			outputs.append(output)

		final_output = tf.stack(outputs, axis=1)
		# print('final_outp	ut', final_output.get_shape())
		return final_output

