import tensorflow as tf
import os
import time
import datetime

from src.text_cnn_embedding import TextCNNEmbedding

"""
iterate over all batches; generator function
"""

def batch_iter(data, scores, batch_size, num_epochs):
    """
    Generates a batch iterator; this is a generator function
    """

    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (data[start_index:end_index],scores[start_index:end_index])

"""
training
"""

def train(flags,cnn_length,no_cnn_batches,data,scores):
    """
    :param flags: hyperparameters; loaded from tf
    :param cnn_length: the length of each input sequence
    :param no_cnn_batches: the number of sequences, i.e., the number of batches to apply cnn to
    :param data: input data
    :param scores: labels or scores
    :return:
    """

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=flags.allow_soft_placement,
          log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            xExpression = tf.placeholder(tf.float32, [None, flags.embedding_dim, 2 * cnn_length * no_cnn_batches], name="input_cnn_x")
            scoresExpression = tf.placeholder(tf.float32, [None], name="scores")

            filters = list(map(int, flags.filter_sizes.split(",")))
            # low level cnn
            cnnLowLevel = TextCNNEmbedding(
                cnn_length=cnn_length,
                no_cnn_batches=2 * no_cnn_batches,
                embedding_size=flags.embedding_dim,
                filter_sizes=filters,
                num_filters=flags.num_filters,
                input=xExpression
            )
            # high level cnn
            cnnHighLevel = TextCNNEmbedding(
                cnn_length=no_cnn_batches,
                no_cnn_batches=2,
                embedding_size=flags.num_filters * len(filters),
                filter_sizes=filters,
                num_filters=flags.num_filters,
                input=cnnLowLevel.output
            )

            # create the loss function
            # norm of the difference
            differenceOfEmbeddings = tf.sub(cnnHighLevel.output[:,:,0],cnnHighLevel.output[:,:,1])
            outputNorm = tf.reduce_sum(tf.pow(differenceOfEmbeddings, 2), 1)
            # sse loss
            lossExpression = tf.reduce_sum(tf.pow(outputNorm-scoresExpression, 2))

            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(lossExpression)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
            # keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)
    
            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
    
            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", lossExpression)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())
    
            # initialize all variables
            sess.run(tf.initialize_all_variables())

            #
            # the actual training
            #

            # generate batches
            batches = batch_iter(data, scores, flags.batch_size, flags.num_epochs)

            for batch in batches:
                feed_dict = {
                    xExpression: batch[0],
                    scoresExpression: batch[1]
                }
                _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, lossExpression],feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

