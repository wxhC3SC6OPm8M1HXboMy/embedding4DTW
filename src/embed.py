import numpy as np
import math
import os

import tensorflow as tf

"""
EMBEDDINGS
"""

def loadInMemoryOneBatch(fileName,batchSize):
    """
    generator function that reads a batch of objects from a file; just one pass on the entire file
    :param fileName: input file name of objects
    :param batchSize: the size of each batch
    :return: list of objects in the batch
    """

    inputFile = open(fileName)

    while True:
        objects = []
        allDone = False
        while True:
            line = inputFile.readline()
            if line:
                objects.append(line)
                if len(objects) == batchSize:
                    break
            else:
                allDone = True
                break
        yield objects
        if allDone == True:
            break

"""
from raw data create feature matrix (we create a copy also)
"""

class CreateFeatureMatrix(object):
    def __init__(self,flags,object2Matrix):
        """
        :param flags: tf flags
        :param object2Matrix: python object that converts an object to a single feature matrix
        """
        
        self.__flags = flags
        self.__object2Matrix = object2Matrix
        
    def createFeatureMatrix(self,batch):
        """
        Given a batch of objects, create the feature matrix by replicating each object and score of 0 (the score can be anything)
        :param batch: input data batch
        :param flags: parameters
        :param object2Matrix: python object to construct a matrix from an input object
        :return:
        """
     
        feature_dim = self.__flags.no_inner_unit * self.__flags.no_outer_unit
        data = np.zeros((len(batch), self.__flags.embedding_dim, 2 * feature_dim), dtype=np.float32)

        count = 0
        for obj in batch:
            m1 = self.__object2Matrix(obj)
            m2 = self.__object2Matrix(obj)
            data[count, :self.__flags.embedding_dim, :feature_dim] = m1
            data[count, :self.__flags.embedding_dim, feature_dim:2 * feature_dim] = m2
            count += 1
            scores = np.zeros(len(batch), dtype=np.float32)

        return (data,scores)

"""
calcuate inference and fetch the actual embedding
"""

class Embedding(object):

    def __init__(self, flags, batch2FeatureMatrixFunc):
        """
        :param flags: tf flags
        :param batch2matrixFunc: function that converts a batch to feature matrix
        """
        self.__flags = flags
        self.__batch2FeatureMatrixFunc = batch2FeatureMatrixFunc

    def batch_iter(self, data, scores, batch_size):
        """
        Generates a batch iterator; this is a generator function

        :param data: tensor or data
        :param scores: numpy 1D array of scores
        :param batch_size: batch size
        """
        
        data_size = len(data)
        num_batches = math.ceil(len(data)/batch_size)
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (data[start_index:end_index],scores[start_index:end_index])

    def evaluate(self, dataBatches):
        """
        Evaluate data which are provided by a generator

        :param pairBatchesGenerator: generator that must return a pair of (data,score); scores can be anything
        """

        print("\nInference! Creating embeddings...\n")

        checkpoint_file = tf.train.latest_checkpoint(self.__flags.checkpoint_dir)
        print(self.__flags.checkpoint_dir)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.__flags.allow_soft_placement,
              log_device_placement=self.__flags.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                data = graph.get_operation_by_name("input_cnn_x").outputs[0]

                # Tensors we want to evaluate: embedding
                embeddings = graph.get_operation_by_name("embedding").outputs[0]

                # Collect embeddings
                allEmbeddings = []
                
                # for all raw data batches
                for dataBatch in dataBatches:
                    dataPair = self.__batch2FeatureMatrixFunc(dataBatch)

                    batches = self.batch_iter(dataPair[0],dataPair[1],self.__flags.batch_size)
                    for batch in batches:
                        feed_dict = {
                            data: batch[0]
                        }
                        batchEmbeddings = sess.run(embeddings, feed_dict)
                        allEmbeddings.append(batchEmbeddings[:,:,0]) # there are two identical copies, we get the first one
                # write the embeddings to file
                for e in allEmbeddings:
                    np.save(self.__flags.embeddings_file,e)

                print("done!\n")

