import tensorflow as tf

"""
Apply one layer of 1-dim CNN + max pooling
Input: matrix of size embedding_size x sequence_length
Output: vector of size num_filters * len(filter_sizes)

Input: self.input
Output: self.output
"""

class TextCNNEmbedding(object):
    """
    sequence_length: the length of the input sequence
    embedding_size: the dimension of the input embedding
    filter_size: list of filter sizes to apply (the largest filter size must be less than or equal to sequence_length
    num_filters: the number of filters to apply
    """
    def __init__(self, sequence_length, embedding_size, filter_sizes, num_filters):

        # place holder for input
        self.input = tf.placeholder(tf.float32, [None, embedding_size, sequence_length], name="input_cnn_x")
        self.input_expanded = tf.expand_dims(self.input, -1)

        # create a convolution plus max pool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution layer
                filter_shape = [embedding_size, filter_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # max pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, sequence_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        self.output = tf.reshape(h_pool, [-1, num_filters_total])



        ###### TO REMOVE
        #outputNorm = tf.reduce_sum(tf.pow(self.output, 2), 1)
        #self.scores = tf.placeholder(tf.float32, [None], name="scores")
        #self.loss = tf.reduce_sum(tf.pow(outputNorm-self.scores, 2))



