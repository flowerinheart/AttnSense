BATCH_SIZE = 64
TRAIN_SIZE = 119080
EVAL_DATA_SIZE = 1193
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))


def run_model(model, train, name):
    data_dir = 'data/sepHARData_a'
    batch_feature, batch_label, batch_eval_feature, batch_eval_label = 
        read_data(os.path.join(datadir, "train"), os.path.join(datadir, "eval"), BATCH_SIZE, '.csv')

    global_step = tf.Variable(0, trainable=False)
    logits = deepSense(batch_feature, train, name)
    predict = tf.argmax(logits, axis=1)

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
    discOptimizer = tf.train.AdamOptimizer(
		learning_rate=1e-4, 
		beta1=0.5,
		beta2=0.9
	).minimize(loss, var_list=t_vars)


    with tf.Session() as sess:
	    tf.global_variables_initializer().run()
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)

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
			    plot.plot('dev accuracy', np.mean(dev_accuracy))
			    plot.plot('dev cross entropy', np.mean(dev_cross_entropy))


		    if (iteration < 5) or (iteration % 50 == 49):
			    plot.flush()

		    plot.tick()





