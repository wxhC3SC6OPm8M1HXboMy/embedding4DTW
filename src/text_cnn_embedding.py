import tensorflow as tf

"""
Apply one layer of 1-dim CNN + max pooling to each batch of the input sequence
Input: tensor matrix of size embedding_size x (cnn_length * no_cnn_batches)
Output: tensor matrix of size (num_filters * len(filter_sizes)) * no_cnn_batches

Input: self.input
Output: self.output
"""

class TextCNNEmbedding(object):
    def __init__(self, gpu, cnn_length, no_cnn_batches, embedding_size, filter_sizes, num_filters, input):
        """
        :param cnn_length: the length of each input sequence
        :param no_cnn_batches: the number of sequences, i.e., the number of batches to apply cnn to
        :param embedding_size: the dimension of the input embedding
        :param filter_sizes: list of filter sizes to apply (the largest filter size must be less than or equal to cnn_length
        :param num_filters: the number of filters to apply
        :param input: input data
        """

        input_expanded = tf.expand_dims(input, -1)

        # create a convolution plus max pool layer for each filter size
        pooled_outputs = []
        with tf.device(gpu):
            for filter_size in filter_sizes:
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # convolution layer
                    filter_shape = [embedding_size, filter_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # for each cnn input batch
                temp_output = []
                for batch in range(no_cnn_batches):
                    portionOfInput = input_expanded[:,:,batch*cnn_length:(batch+1)*cnn_length,:]

                    conv = tf.nn.conv2d(
                        portionOfInput,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # max pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 1, cnn_length - filter_size + 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

        # combine all the pooled features
        # first output coming from the same cnn batch
        temp = [[] for _ in range(no_cnn_batches)]
        for i,x in enumerate(pooled_outputs):
            temp[i % no_cnn_batches].append(x)
        temp1 = []
        for x in temp:
            temp1.append(tf.concat(3, x))
        # create a matrix
        result = tf.concat(2,temp1)
        result = tf.transpose(result,perm=[0,1,3,2])

        self.output = tf.squeeze(result)

