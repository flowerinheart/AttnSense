import os
import tensorflow as tf 

def read_audio_csv(filename_queue, wide, feature_dim, out_dim):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
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

def get_file_list(data_dir, suffix):
	fileList = []
	orgFileList = os.listdir(data_dir)
	for tfile in orgFileList:
		if tfile.endswith(suffix):
			fileList.append(os.path.join(data_dir, tfile))
	return fileList

def read_data(train_path, eval_path, batch_size, suffix):
    trainFileList = get_file_list(train_path, suffix)
    evalFileList = get_file_list(eval_path, suffix)
    batch_feature, batch_label = input_pipeline(trainFileList, batch_size)
    batch_eval_feature, batch_eval_label = input_pipeline(evalFileList, batch_size, shuffle_sample=False)
    return batch_feature, batch_label, batch_eval_feature, batch_eval_label
