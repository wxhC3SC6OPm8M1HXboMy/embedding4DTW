import pickle
import os

import src.parameters as params
import src.embed as embed

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

for each sentence or paragraph, the function outputs the corresponding embedding
input sentences/paragraphs are read from input in large batches loaded to memory, and then smaller batches are processed for embedding
"""

CHARACTER_FILE = "data/chars.pkl"

# if empty, then the most recent one is taken; if non-empty, then the relative path based on curdir is taken
params.tf.flags.DEFINE_string("checkpoint_abs_dir", "", "Checkpoint non-default directory")

"""
EMBED
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
        setDir = os.path.abspath(os.path.join(os.path.curdir, setDir))

    flags.checkpoint_dir = setDir

def main_embed():

    # set all hyper parameters
    flags = params.setParameters()

    # set the checkpoint directory
    checkpointDir(flags)

    # read the mapping of characters to array indices
    character_dict = pickle.load( open( CHARACTER_FILE, "rb" ) )

    # read embedding data; generator
    batches = embed.loadInMemoryOneBatch(flags.embed_text_file, flags.embed_batch_size)

    # classes for creating the matrix from sentences
    sentence2Matrix = dataprep.CreateMatrixFromSentence(character_dict,flags.no_inner_unit,flags.no_outer_unit)
    data2Matrix = embed.CreateFeatureMatrix(flags, sentence2Matrix.createMatrix)

    # peform the actual embeddings
    embedObj = embed.Embedding(flags,data2Matrix.createFeatureMatrix)
    embedObj.evaluate(batches)

main_embed()
