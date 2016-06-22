import tensorflow as tf
import numpy as np
import math

class Evaluate(object):

    def __init__(self, flags):
        """
        :param flags: tf flags
        """
        self.__flags = flags

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

    def evaluate(self, pairBatchesGenerator):
        """
        Evaluate test data which are provided by a generator

        :param pairBatchesGenerator: generator that must return a pair of (data,score)
        """

        print("\nEvaluating on test data...\n")

        checkpoint_file = tf.train.latest_checkpoint(self.__flags.checkpoint_dir)
        print(self.__flags.checkpoint_dir)
        print(checkpoint_file)
        print(x)
        graph = tf.Graph()
        errors = []
        trueScores = []
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

                # Tensors we want to evaluate
                scores = graph.get_operation_by_name("output/scores").outputs[0]

                # Collect the errors
                for pairBatch in pairBatchesGenerator:
                    batches = self.batch_iter(pairBatch[0],pairBatch[1],self.__flags.batch_size)
                    for batch in batches:
                        feed_dict = {
                            data: batch[0]
                        }
                        batchScores = sess.run(scores, feed_dict)
                        errors = np.concatenate([errors, batch[1]-batchScores])
                        trueScores = np.concatenate([trueScores, batch[1]])

            # print metrics
            print("Total number of test examples: {}".format(len(errors)))
            avgSSE = sum([x*x for x in errors])/float(len(errors))
            print("Averate SSE: {:g}".format(avgSSE))
            # when true score is 0, error is also zero since the two objects are identical
            mape = sum([abs(x[0])/x[1] for x in zip(errors,trueScores) if x[1] > self.__flags.tolerance])/float(len(errors))
            print("MAPE: {:g}".format(mape))
