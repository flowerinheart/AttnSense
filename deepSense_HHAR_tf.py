# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

import plot

import time
import math
import os
import sys
from tensorflow.python.layers.core import Dropout, Dense

layers = tf.contrib.layers

#假设输入是W, W={Vt, Ut},Vt大小事n * d, T是W的大小，Ut是t， 对于每个Vt, 使用傅里叶变换将W变换成d * 2 * f * t
SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*6*2
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 64
TOTAL_ITER_NUM = 30000

select = 'a'

metaDict = {'a':[119080, 1193], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))

###### Import training data
def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
		#每一条输入在预处理的时候整理成了WIDE*FEATURE_DIM+OUT_DIM的长度
	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + OUT_DIM)]

	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])
	labels = fileData[WIDE*FEATURE_DIM:]
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1000#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch

######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True,
# 			updates_collections=None, scope=scope),
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True))
def attention_fusion_1(inputs):
	"""inputs, shape: [batch, time_step, input_mode, feature_dim]"""
	d = inputs.shape.as_list()[3]
	w = tf.Variable(tf.random_normal([d], stddev=0.1))
	b = tf.Variable(tf.random_normal([]))
	#activation = tf.tanh(tf.matmul(inputs, w) + b)  # b * t *  i
	activation = tf.tanh(tf.tensordot(inputs, w, axes=1) + b)  # b * t *  i
	alphas = tf.nn.softmax(activation)  # b * t * i
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

	with tf.name_scope('v'):
		# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
		#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
		v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

	# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
	vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
	alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

	# Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
	output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

	if not return_alphas:
		return output
	else:
		return output, alphas

def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True,
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)



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
            u_w = tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.01, seed=1234))
            x = tf.tensordot(x_proj, u_w, axes=1)
            alphas = tf.nn.softmax(x, axis=-1)
            output = tf.matmul(tf.transpose(t_x, [0, 2, 1]), alphas)
            output = tf.squeeze(output, -1)
            outputs.append(output)

        final_output = tf.stack(outputs, axis=1)
        # print('final_outp	ut', final_output.get_shape())
        return final_output

def sensor_local_feature_layer(inputs, name, train):
		conv1 = layers.convolution2d(inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
						stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope=(name + '_conv1'))# CONV_NUM 64, filter number
		conv1 = batch_norm_layer(conv1, train, scope=(name + '_BN1'))#output is [64, 20, 8, 64], CONV_NUM is 64
		conv1 = tf.nn.relu(conv1)
		conv1_shape = conv1.get_shape().as_list()
		print("conv1 " + str(conv1_shape))
		conv1 = layers.dropout(conv1, CONV_KEEP_PROB, is_training=train,
			noise_shape=[conv1_shape[0], 1, 1, conv1_shape[3]], scope=name+'_dropout1')

		conv2 = layers.convolution2d(conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE], #3
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope=name+'_conv2')
		conv2 = batch_norm_layer(conv2, train, scope=name+'_BN2')
		conv2 = tf.nn.relu(conv2)
		conv2_shape = conv2.get_shape().as_list()
		print("conv2 " + str(conv2_shape))
		conv2 = layers.dropout(conv2, CONV_KEEP_PROB, is_training=train,
			noise_shape=[conv2_shape[0], 1, 1, conv2_shape[3]], scope=name+'_dropout2')

		conv3 = layers.convolution2d(conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST], #3
						stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope=name+'_conv3')
		conv3 = batch_norm_layer(conv3, train, scope=name+'_BN3')
		conv3 = tf.nn.relu(conv3)
		conv3_shape = conv3.get_shape().as_list()
		print("conv3 " + str(conv3_shape))

		conv_out = tf.reshape(conv3, [conv3_shape[0], conv3_shape[1], 1, conv3_shape[2], conv3_shape[3]])
		return conv_out

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)

		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		# input是20 * 120, 20是时间窗口大小，60是 6 * 2 * f, 傅里叶变换后的
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)
		print("input " + str(acc_inputs.shape))

		acc_conv_out = sensor_local_feature_layer(acc_inputs, "acc", train)
		gyro_conv_out = sensor_local_feature_layer(gyro_inputs, "gyro", train)


		sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		print("sensor_conv_in: " + str(sensor_conv_in.shape))
		sensor_conv3 = sensor_conv_in
		sensor_conv3_shape = sensor_conv3.get_shape().as_list()

		print("sensor conv3 shape " + str(sensor_conv3_shape))
		#sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]*sensor_conv3_shape[4]])

	 	attention_input = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2], sensor_conv3_shape[3]*sensor_conv3_shape[4]])
		print("attention fusion input " + str(attention_input.shape))
		sensor_conv_out = attention_fusion_1(attention_input)
		#self attention fusion
		#sensor_conv_out = self_attention(attention_input, "self_attention")
		print("attention fusion output " + str(sensor_conv_out.shape))



		gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
		if train:
			gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

		gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
		if train:
			gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

		cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
		init_state = cell.zero_state(BATCH_SIZE, tf.float32)

		cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)
		print("debug: rnn_output:" + str(cell_output.shape))#64,20,120
		print("mask: rnn_output:" + str(mask.shape))

		AZ = 80
		attention_out = attention(cell_output, attention_size=AZ)
		print("debug: attention_out:" + str(attention_out.shape) + " " + str(AZ))
		avg_cell_out = attention_out

		# average of all cell output
		#sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		#avg_cell_out = sum_cell_out/avgNum

		#sum_cell_out = tf.reduce_sum(cell_output, axis=1, keep_dims=False)
		#avg_cell_out = sum_cell_out
		#print("debug: sum_cell_out:" + str(sum_cell_out.shape))

		logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		print("logits " + str(logits.shape))
		return logits

csvFileList = []
csvDataFolder1 = os.path.join('sepHARData_'+select, "train")
orgCsvFileList = os.listdir(csvDataFolder1)
for csvFile in orgCsvFileList:
	if csvFile.endswith('.csv'):
		csvFileList.append(os.path.join(csvDataFolder1, csvFile))

csvEvalFileList = []
csvDataFolder2 = os.path.join('sepHARData_'+select, "eval")
orgCsvFileList = os.listdir(csvDataFolder2)
for csvFile in orgCsvFileList:
	if csvFile.endswith('.csv'):
		csvEvalFileList.append(os.path.join(csvDataFolder2, csvFile))

global_step = tf.Variable(0, trainable=False)

batch_feature, batch_label = input_pipeline(csvFileList, BATCH_SIZE)
batch_eval_feature, batch_eval_label = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)

# train_status = tf.placeholder(tf.bool)
# trainX = tf.cond(train_status, lambda: tf.identity(batch_feature), lambda: tf.identity(batch_eval_feature))
# trainY = tf.cond(train_status, lambda: tf.identity(batch_label), lambda: tf.identity(batch_eval_label))

# logits = deepSense(trainX, train_status, name='deepSense')
logits = deepSense(batch_feature, True, name='deepSense')


predict = tf.argmax(logits, axis=1)

# batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trainY)
batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
loss = tf.reduce_mean(batchLoss)

logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
predict_eval = tf.argmax(logits_eval, axis=1)
loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))

t_vars = tf.trainable_variables()

regularizers = 0.
for var in t_vars:
	regularizers += tf.nn.l2_loss(var)
loss += 5e-4 * regularizers

# optimizer = tf.train.RMSPropOptimizer(0.001)
# gvs = optimizer.compute_gradients(loss, var_list=t_vars)
# capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
# discOptimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)

discOptimizer = tf.train.AdamOptimizer(
		learning_rate=1e-4,
		beta1=0.5,
		beta2=0.9
	).minimize(loss, var_list=t_vars)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	best = -0.1
	for iteration in xrange(TOTAL_ITER_NUM):

		# _, lossV, _trainY, _predict = sess.run([discOptimizer, loss, trainY, predict], feed_dict = {
		# 	train_status: True
		# 	})
		_, lossV, _trainY, _predict = sess.run([discOptimizer, loss, batch_label, predict])
		_label = np.argmax(_trainY, axis=1)
		_accuracy = np.mean(_label == _predict)
		plot.plot('train cross entropy', lossV)
		plot.plot('train accuracy', _accuracy)


		if iteration % 50 == 49:
			dev_accuracy = []
			dev_cross_entropy = []
			for eval_idx in xrange(EVAL_ITER_NUM):
				# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
				eval_loss_v, _trainY, _predict = sess.run([loss, batch_eval_label, predict_eval])
				_label = np.argmax(_trainY, axis=1)
				_accuracy = np.mean(_label == _predict)
				dev_accuracy.append(_accuracy)
				dev_cross_entropy.append(eval_loss_v)
			best = max(best, np.mean(dev_accuracy))
			plot.plot('dev accuracy', np.mean(dev_accuracy))
			plot.plot('dev cross entropy', np.mean(dev_cross_entropy))


		if (iteration < 5) or (iteration % 50 == 49):
			plot.flush()

		plot.tick()
	print("best score " + str(best))
