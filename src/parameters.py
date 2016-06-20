import tensorflow as tf

def setParameters():

    """
    Application specific (should probably not be here)
    """
    # applicable for sentence to embedding
    tf.flags.DEFINE_string("distance_type", "lev", "Distance type") # must be lev or word2vec
    tf.flags.DEFINE_string("word2vec_folder", "../data/word2vec", "Word2vec folder")

    """
    Data parameters
    """
    tf.flags.DEFINE_string("train_text_file", "data/train_set_data_sentences.txt", "Train input file name")
    tf.flags.DEFINE_string("validate_text_file", "data/validate_set_data_sentences.txt", "Validate input file name")
    tf.flags.DEFINE_string("test_text_file", "data/test_set_data_sentences.txt", "Test input file name")
    # the number of objects to load to memory at once
    tf.flags.DEFINE_integer("batch_size_to_load_to_memory", 50000, "Number of text objects to load to memory")
    # the number of times we load objects to memory
    tf.flags.DEFINE_integer("no_epochs", 12, "Number of epochs")

    """
    Input data embedding
    """
    tf.flags.DEFINE_integer("embedding_dim", 61, "Dimensionality of low level embedding")
    tf.flags.DEFINE_integer("no_inner_unit", 26, "Max number of inner units")
    tf.flags.DEFINE_integer("no_outer_unit", 51, "Max number of outer units")
    tf.flags.DEFINE_integer("batch_size_per_memory_batch", 100, "Batch size per memory batch")
    tf.flags.DEFINE_integer("no_in_memory_pair_batches_to_process_per_memory_epoch", 50, "Number of in_memory pair batches to process per memory epoch")
    tf.flags.DEFINE_integer("max_pairs_of_batches_per_memory_epoch", 5, "Maximum number of pairs of in-memory batches per in memory epoch")

    """
    Model hyperparameters
    """
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

    """
    Training parameters
    """
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

    """
    Validation and test parameters
    """
    tf.flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory")
    tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("validation_batch_size", 500, "Batch size for creating pairs of objects for validation")
    tf.flags.DEFINE_integer("test_batch_size", 500, "Batch size for creating pairs of objects for evaluation")

    """
    Misc parameters
    """
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    tf.flags.DEFINE_integer("seed", 38761453, "Seed for random generator")
    tf.flags.DEFINE_string("gpu", "/gpu:1", "Gpu used")
    tf.flags.DEFINE_float("tolerance", 0.0000001, "Floating point tolerance")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    return FLAGS
