import pickle
import numpy as np
import tensorflow as tf

import src.parameters as params
import src.train as train

"""
lowLevel
highLevel

for sentence embedding, we have: lowLevel = characters; highLevel = words
    NUM_INNER_UNIT = number of characters per word; NUM_OUTER_UNIT = number of words per sentence
for paragraph embedding, we have: lowLevel = words; highLevel = sentence
    NUM_INNER_UNIT = number of words per sentence; NUM_OUTER_UNIT = number of sentences per paragraph
"""

OBSERVATION_FILE = "../data/data.pkl"
SCORE_FILE = "../data/label.pkl"

NUM_INNER_UNIT = 15
NUM_OUTER_UNIT = 12

# read the data
data = pickle.load( open(OBSERVATION_FILE,"rb"))
numberOfObservations = data.shape[0]
lowLevelEmbeddingDim = data.shape[1]
lowLevelEmbeddingFeatureDim = data.shape[2]

dataScores = pickle.load( open(SCORE_FILE,"rb"))

# check if the size matches
if 2 * NUM_INNER_UNIT * NUM_OUTER_UNIT + 2 * (NUM_OUTER_UNIT-1) != lowLevelEmbeddingFeatureDim:
    print("Error: Dimension mismatch in the number of features in the input data!")
    exit(1)

# check if observations and labels match
if len(dataScores) != data.shape[0]:
    print("Error: Dimension mismatch in the number of scores and observations!")
    exit(1)

dataScores = np.array(list(map(float, dataScores)))

# set all hyper parameters
flags = params.setParameters(lowLevelEmbeddingDim)

# train the model
train.train(flags,lowLevelEmbeddingFeatureDim,data,dataScores)