from layer import common

def sensor_local_conv(inputs, train, name):
    conv1 = common.conv_with_dropout(inputs, name=name+"_1", train, 64, kernel_size=[1, 18], 
		strides=[1, 6], padding='VALID', data_format='NHWC', dropout_prob = 0.8)
    conv2 = common.conv_with_dropout(conv1, name=name+"_2", train, 64, kernel_size=[1, 3], 
		strides=[1, 1], padding='VALID', data_format='NHWC', dropout_prob = 0.8)
    conv3 = common.conv(conv2, name=name+"_3", train, 64, kernel_size=[1, 3], strides=[1, 1], 
		padding='VALID', data_format='NHWC')
    conv3_shape = conv3.get_shape().as_list()
    conv_out = tf.reshape(conv3, [conv3_shape[0], conv3_shape[1], 1, conv3_shape[2], conv3_shape[3]])
    return conv_out

def merge_conv(inputs, train, name):
    conv1 = common.conv_with_dropout(inputs, name=name+"_1", train, 64, kernel_size=[1, 2, 8], 
		strides=[1, 1, 1], padding='SAME', data_format='NDHWC', dropout_prob = 0.8)
    conv2 = common.conv_with_dropout(conv1, name=name+"_2", train, 64, kernel_size=[1, 2, 6], 
		strides=[1, 1, 1], padding='SAME', data_format='NDHWC', dropout_prob = 0.8)
    conv3 = common.conv(conv2, name=name+"_3", train, 64, kernel_size=[1, 2, 4], strides=[1, 1], 
		padding='SAME`', data_format='NDHWC')
    conv3_shape = conv3.get_shape().as_list()
    conv_out = tf.reshape(conv3, [conv3_shape[0], conv3_shape[1], conv3_shape[2] * conv3_shape[3] * conv3_shape[4]])
    return conv_out

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse) as scope:
        BATCH_SIZE = inputs.shape[0]
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)


		# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
		# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
		sensor_inputs = tf.expand_dims(inputs, axis=3)
		acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)
        acc_conv_out = sensor_local_conv(acc_inputs, train, "acc_conv")
        gyro_conv_out = sensor_local_conv(gyro_inputs, train, "gyro_conv")

        sensor_conv_in = tf.concat([acc_conv_out, gyro_conv_out], 2)
		senor_conv_shape = sensor_conv_in.get_shape().as_list()	
		sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
			noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], scope='sensor_dropout_in')

        sensor_conv_out = merge_conv(sensor_conv_in, train, "merge_conv")

        cell_output, final_stateTuple = bi_gru(sensor_conv_out, 120, length, BATCH_SIZE, train)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)
		sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
		avg_cell_out = sum_cell_out/avgNum


        logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits


# def merge_conv_without_dropout(inputs, name, train, conv_num, kernel_size, strides):
# 	conv = layers.convolution2d(inputs, conv_num, kernel_size=kernel_size,
# 			stride=strides, padding='SAME', activation_fn=None, data_format='NDHWC', scope=name+"_conv")
#     conv = batch_norm_layer(conv, train, scope=name+'_BN')
# 	conv = tf.nn.relu(conv)
#     return conv

# def merge_conv_with_dropout(inputs, name, train, conv_num, kernel_size, strides, dropout_prob):
#     conv = merge_conv_without_dropout(inputs, name, train, conv_num, kernel_size, strides)
# 	conv_shape = conv.get_shape().as_list()
# 	conv = layers.dropout(conv, dropout_prob, is_training=train,
# 			noise_shape=[conv_shape[0], 1, 1, 1, conv_shape[4]], scope=name+'_dropout')
#     return conv

# def merge_conv(inputs, train, name):
#     conv1 = common.merge_conv_with_dropout(inputs, name=name+"_1", train, 64, kernel_size=[1, 2, 8], strides=[1, 1, 1], dropout_prob = 0.8)
#     conv2 = common.merge_conv_with_dropout(conv1, name=name+"_2", train, 64, kernel_size=[1, 2, 6], strides=[1, 1, 1], dropout_prob = 0.8)
#     conv3 = common.merge_conv_without_dropout(conv2, name=name+"_3", train, 64, kernel_size=[1, 2, 4], strides=[1, 1, 1])
# 	conv3_shape = conv3.get_shape().as_list()
# 	conv_out = tf.reshape(conv3, [conv3_shape[0], conv3_shape[1], conv3_shape[2]*conv3_shape[3]*conv3_shape[4]])
#     return conv_out

