import tensorflow as tf
import os
import time
import datetime

from src.text_cnn_embedding import TextCNNEmbedding

"""
training
"""

def train(flags,sequence_length,data,scores):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=flags.allow_soft_placement,
          log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNEmbedding(
                sequence_length=sequence_length,
                embedding_size=flags.embedding_dim,
                filter_sizes=list(map(int, flags.filter_sizes.split(","))),
                num_filters=flags.num_filters)
    
            # define training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
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
            loss_summary = tf.scalar_summary("loss", cnn.loss)

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

            feed_dict = {
                cnn.input: data,
                cnn.scores: scores
            }
            _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, cnn.loss],feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)

