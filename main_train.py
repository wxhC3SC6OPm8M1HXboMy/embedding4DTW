import pickle
import random

import src.parameters as params
import src.train as train

import tools.prep_data as dataprep

"""
lowLevel
highLevel

for sentence embedding, we have: lowLevel = characters; highLevel = words
    NO_INNER_UNIT = number of characters per word; NO_OUTER_UNIT = number of words per sentence (these values must be set in parameters)
    object = sentence
for paragraph embedding, we have: lowLevel = words; highLevel = sentence
    NO_INNER_UNIT = number of words per sentence; NO_OUTER_UNIT = number of sentences per paragraph (these values must be set in parameters)
    object = paragraph

train data is loaded and prepared in batches (everything is batch by batch with nesting)
validation data is loaded at once (as one batch) and prepared in batches (due to potentially big size of all pairs of objects)
"""

CHARACTER_FILE = "data/chars.pkl"

"""
# to compute the basic parameters; this function should not be called
flags = params.setParameters()
(chars,max_char_in_word, max_words_in_sentence) = dataprep.computeParameters(flags.text_file)
print(max_char_in_word)
print(max_words_in_sentence)
chars.append('\t')
x = dict()
for idx,v in enumerate(character_list):
    x[v] = idx
pickle.dump(x, open(CHARACTER_FILE, "wb" ))
exit(1)
"""

"""
TRAIN
"""

def main_train():

    # set all hyper parameters
    flags = params.setParameters()

    random.seed(flags.seed)

    # read the mapping of characters to array indices
    character_dict = pickle.load( open( CHARACTER_FILE, "rb" ) )

    # read validation data
    validationData = train.loadAllData(flags.validate_text_file)

    # class for creating the matrix from sentences
    sentence2Matrix = dataprep.CreateMatrixFromSentence(character_dict,flags.no_inner_unit,flags.no_outer_unit)

    # class for computing the distance between two sentences
    distanceObj = dataprep.ComputeDistance(flags.distance_type,flags.word2vec_folder)

    # data validation processing class
    processOneValidationBatch = train.ProcessInMemoryBatch(sentence2Matrix.createMatrix,distanceObj.computeDistance,
                                                           flags.embedding_dim,flags.no_inner_unit * flags.no_outer_unit)    

    # the training object
    training = train.Train(flags.no_inner_unit, flags.no_outer_unit, flags)
    with training:
        training.buildModel()

        # one by one load batches of objects to memory and process each one of them
        batch = train.loadInMemoryOneBatch(flags.train_text_file,flags.batch_size_to_load_to_memory)
        for i in range(flags.no_epochs):
            print("Loading batch %d of objects to memory!" % (i+1))
            # for each in memory batch, slice into smaller batches and then create pairs of objects for training
            processOneBatch = train.ProcessInMemoryBatch(sentence2Matrix.createMatrix,distanceObj.computeDistance,
                                                         flags.embedding_dim,flags.no_inner_unit * flags.no_outer_unit)
            pairBatch = processOneBatch.process(next(batch),flags.batch_size_per_memory_batch,flags.max_pairs_of_batches_per_memory_epoch)
            # execute epochs
            for j in range(flags.no_in_memory_pair_batches_to_process_per_memory_epoch):
                # execute a single pair of batches
                print("Processing pair batch number %d" % (j+1))
                (data,scores) = next(pairBatch)
                # validation generator
                validationDataGenerator = processOneValidationBatch.process(validationData,flags.validation_batch_size,noPasses=1)

                # actual training
                training.train(data, scores, validationDataGenerator)

main_train()

