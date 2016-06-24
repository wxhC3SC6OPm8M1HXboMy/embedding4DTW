import pickle
import os

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

# if empty, then the most recent one is taken; if non-empty, then the relative path based on curdir is taken
params.tf.flags.DEFINE_string("checkpoint_abs_dir", "", "Checkpoint non-default directory")

"""
EVALUATE
"""

def checkpointDir(flags):
    setDir = flags.checkpoint_abs_dir
    if setDir == "":
        # fetch the most recent directory in runs
        dirName = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        dirs = [os.path.join(dirName,d) for d in os.listdir(dirName)]
        lastDir = max([d for d in dirs if os.path.isdir(d)], key = os.path.getmtime)
        setDir = os.path.abspath(os.path.join(lastDir, flags.checkpoint_dir))
    else:
        setDir = os.path.abspath(os.path.join(os.path.curdir, setDir, flags.checkpoint_dir))
    print(setDir)
    flags.checkpoint_dir = setDir

def main_evaluate():

    # set all hyper parameters
    flags = params.setParameters()

    # set the checkpoint directory
    checkpointDir(flags)

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

    pairBatches = processOneBatch.process(validationData, flags.test_batch_size,noPasses=1)

    # evaluate on test data
    evaluateObj = evaluate.Evaluate(flags)
    evaluateObj.evaluate(pairBatches)

main_evaluate()

