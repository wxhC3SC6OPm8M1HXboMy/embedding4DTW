import pickle

import src.parameters as params
import src.train as train

import tools.prep_data as dataprep
import src.evaluate as evaluate

"""
lowLevel
highLevel

for sentence embedding, we have: lowLevel = characters; highLevel = words
    NO_INNER_UNIT = number of characters per word; NO_OUTER_UNIT = number of words per sentence (these values must be set in parameters)
    object = sentence
for paragraph embedding, we have: lowLevel = words; highLevel = sentence
    NO_INNER_UNIT = number of words per sentence; NO_OUTER_UNIT = number of sentences per paragraph (these values must be set in parameters)
    object = paragraph

test data is loaded at once (as one batch) and prepared in batches (due to potentially big size of all pairs of objects)
"""

CHARACTER_FILE = "data/chars.pkl"

"""
EVALUATE
"""

def main_evaluate():

    # set all hyper parameters
    flags = params.setParameters()

    # read the mapping of characters to array indices
    character_dict = pickle.load( open( CHARACTER_FILE, "rb" ) )

    # read test data
    validationData = train.loadAllData(flags.test_text_file)

    # class for creating the matrix from sentences
    sentence2Matrix = dataprep.CreateMatrixFromSentence(character_dict,flags.no_inner_unit,flags.no_outer_unit)

    # class for computing the distance between two sentences
    distanceObj = dataprep.ComputeDistance(flags.distance_type,flags.word2vec_folder)

    # iterate through all batches
    processOneBatch = train.ProcessInMemoryBatch(sentence2Matrix.createMatrix, distanceObj.computeDistance,
                                                 flags.embedding_dim, flags.no_inner_unit * flags.no_outer_unit)

    pairBatches = processOneBatch.process(validationData, flags.test_batch_size)

    # evaluate on test data
    evaluateObj = evaluate.Evaluate(flags)
    evaluateObj.evaluate(pairBatches)

main_evaluate()

