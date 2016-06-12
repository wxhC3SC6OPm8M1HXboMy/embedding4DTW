import tensorflow as tf
import os
import time
import datetime
import random
import numpy as np

from src.text_cnn_embedding import TextCNNEmbedding

"""
CREATE A FEED OF RAW DATA (objects) IN BATCHES; EACH BATCH READY TO BE PROCESSED BY TENSORFLOW FOR TRAINING
"""

"""
load batches from file to memory as list
"""

def loadInMemoryOneBatch(fileName,batchSize):
    """
    generator function that reads a batch of sentences from a file
    :param fileName: input file name of sentences
    :param batchSize: the size of each batch
    :return: list of sentences in the batch
    """

    inputFile = open(fileName)

    while True:
        sentences = []
        while True:
            line = inputFile.readline()
            if line:
                sentences.append(line)
                if len(sentences) == batchSize:
                    break
            else:
                inputFile.seek(0)
        yield sentences

"""
processes one in-memory batch

Example given an in-memory batch b of size 100, batch_size = 15, max_pairs = 3
create ceil(100/15) = 7 batches from b; each one of size 15 except for the last one
create all possible pairs s = {(1,1),(1,2),...,(7,7)}
randomly shuffle s into t
indefinitely loop in cyclic way through s each time considering only the first 3 pairs

If random shuffle is (1,5),(2,6),(1,1),etc, then the identifine loop is:
(1,5),(2,6),(1,1),(1,5),(2,6),(1,1),(1,5),(2,6),(1,1),...

Given a pair of batches (x,y), generate all pairs of objects, compute the input embedding matrix, and the score
"""

class ProcessInMemoryBatch(object):
    """
    For processing one batch of objects that are stored in memory
    """

    def __init__(self,object2MatrixFunction,distanceBetween2Objects,embedding_dim,feature_dim):
        """
        :param object2MatrixFunction: function that takes an object and creates an embedding as 2D numpy array of the object
        :param distanceBetween2Objects: function that takes two objects and returns the distance
        :param embedding_dim: the dimension of raw object embedding
        :param feature_dim: the number of features per object
        """
        self.__object2MatrixFunction = object2MatrixFunction
        self.__distanceBetween2Objects = distanceBetween2Objects

        self.__embedding_dim = embedding_dim
        self.__feature_dim = feature_dim

    def __createAllPairs(self,batch1,batch2):
        """
        Scan all pairs and crate an input matrix of pairs
        :param batch1: first batch
        :param batch2: second batch
        :return: pair of: 3D numpy tensor; dim 0 = instance, dim 1 = embedding, dim 2 = features AND 1D numpy tensor of scores
        """

        scores = np.zeros(len(batch1)*len(batch2),dtype=np.float32)
        data = np.zeros((len(batch1)*len(batch2),self.__embedding_dim, 2 * self.__feature_dim),dtype=np.float32)
        count = 0
        for obj1 in batch1:
            for obj2 in batch2:
                # first create the data matrix
                m1 = self.__object2MatrixFunction(obj1)
                m2 = self.__object2MatrixFunction(obj2)
                data[count, :self.__embedding_dim, :self.__feature_dim] = m1
                data[count, :self.__embedding_dim, self.__feature_dim:2 * self.__feature_dim] = m2
                # compute the distance
                dist = self.__distanceBetween2Objects(obj1,obj2)
                scores[count] = dist
                count += 1

        return (data,scores)

    def process(self,corpus,batch_size,max_pairs = None):
        """
        Main generator function that processes one batch of objects loaded into memory
        All objects are divided into batches
        Then all pairs of batches are created and sorted randomly
        The pairs are then processed as mini epochs; For each pair all pairs of objects are craeted and fed into batch_iter

        :param corpus: list of objects
        :param batch_size: batch size from objects in corpus
        :return pair of: 3D numpy tensor; dim 0 = instance, dim 1 = embedding, dim 2 = features AND 1D numpy tensor of scores
        """

        # craete all pairs and ranodomly permute them
        num_batches_per_epoch = int(len(corpus) / batch_size) + 1
        allPairs = [(i,j) for i in range(num_batches_per_epoch) for j in range(i,num_batches_per_epoch)]
        random.shuffle(allPairs)

        if max_pairs == None:
            max_pairs = len(allPairs)
        elif max_pairs > len(allPairs):
            max_pairs = len(allPairs)
            print("max_pairs in process exceeds the size of the list of all pairs of batches! Resetting to the length of the list!")

        n = len(corpus)

        # iterate forever
        count = 0 # index into allPairs
        while True:
            t = allPairs[count]
            from1, to1 = (t[0] * batch_size, min((t[0]+1) * batch_size, n))
            from2, to2 = (t[1] * batch_size, min((t[1]+1) * batch_size, n))
            yield self.__createAllPairs(corpus[from1:to1], corpus[from2:to2])
            count = ((count+1) % max_pairs)

"""
MINI-BATCH SGD IN TENSORFLOW
"""

class Train(object):

    def __init__(self,no_inner_units,no_outer_units,flags):
        """
        :param no_inner_units: number of inner units
        :param no_outer_units: number of outer units
        :param flags: hyperparameters; loaded from tf
        """

        self.__cnn_length = no_inner_units
        self.__no_cnn_batches = no_outer_units

        self.__flags = flags

    def __enter__(self):
        tf.Graph().as_default().__enter__()
        tf.device('/cpu:0').__enter__()

    def __exit__(self,*args):
        tf.Graph().as_default().__exit__(*args)
        tf.device('/cpu:0').__exit__(*args)

    def train(self, data, scores):
        """
        The actual training

        :param data: data
        :param scores: scores
        """

        # generate batches
        batches = self.batch_iter(data, scores, self.__flags.batch_size, self.__flags.num_epochs)

        for batch in batches:
            feed_dict = {
                self.__xExpression: batch[0],
                self.__scoresExpression: batch[1]
            }
            _, step, summaries, loss = self.__sess.run([self.__train_op, self.__global_step, self.__train_summary_op,
                                                        self.__lossExpression], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            self.__train_summary_writer.add_summary(summaries, step)

    def batch_iter(self,data, scores, batch_size, num_epochs):
        """
        Generates a batch iterator; this is a generator function

        :param scores: numpy 1D array of scores
        :param batch_size: batch size in SGD
        :param num_epochs: number of epochs
        """

        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield (data[start_index:end_index],scores[start_index:end_index])

    def buildModel(self):
        """
        The actual training

        :param cnn_length: the length of each input sequence
        :param no_cnn_batches: the number of sequences, i.e., the number of batches to apply cnn to
        :param data: input data
        :param scores: labels or scores
        """

        tf.Graph().as_default()
        tf.device('/cpu:0')

        # with tf.Graph().as_default(),tf.device('/cpu:0'):
        session_conf = tf.ConfigProto(
          allow_soft_placement=self.__flags.allow_soft_placement,
          log_device_placement=self.__flags.log_device_placement)
        self.__sess = tf.Session(config=session_conf)
        with self.__sess.as_default():
            self.__xExpression = tf.placeholder(tf.float32, [None, self.__flags.embedding_dim, 2 * self.__cnn_length * self.__no_cnn_batches], name="input_cnn_x")
            self.__scoresExpression = tf.placeholder(tf.float32, [None], name="scores")

            filters = list(map(int, self.__flags.filter_sizes.split(",")))
            # low level cnn
            cnnLowLevel = TextCNNEmbedding(
                gpu=self.__flags.gpu,
                cnn_length=self.__cnn_length,
                no_cnn_batches=2 * self.__no_cnn_batches,
                embedding_size=self.__flags.embedding_dim,
                filter_sizes=filters,
                num_filters=self.__flags.num_filters,
                input=self.__xExpression
            )
            # high level cnn
            cnnHighLevel = TextCNNEmbedding(
                gpu=self.__flags.gpu,
                cnn_length=self.__no_cnn_batches,
                no_cnn_batches=2,
                embedding_size=self.__flags.num_filters * len(filters),
                filter_sizes=filters,
                num_filters=self.__flags.num_filters,
                input=cnnLowLevel.output
            )

            # create the loss function
            # norm of the difference
            differenceOfEmbeddings = tf.sub(cnnHighLevel.output[:,:,0],cnnHighLevel.output[:,:,1])
            outputNorm = tf.reduce_sum(tf.pow(differenceOfEmbeddings, 2), 1)
            # sse loss
            self.__lossExpression = tf.reduce_sum(tf.pow(outputNorm-self.__scoresExpression, 2))

            # define training procedure
            self.__global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.__lossExpression)
            self.__train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.__global_step)

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
            loss_summary = tf.scalar_summary("loss",self.__lossExpression)

            # Train Summaries
            self.__train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            self.__train_summary_writer = tf.train.SummaryWriter(train_summary_dir, self.__sess.graph)

            # checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            tf.train.Saver(tf.all_variables())

            # initialize all variables
            self.__sess.run(tf.initialize_all_variables())
