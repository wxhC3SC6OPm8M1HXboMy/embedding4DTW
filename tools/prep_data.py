import numpy as np
from gensim.models import Word2Vec
import tools.distance_DTW as dtw

def computeParameters(fileName):
    """
    This function should never be called on a very large input file
    Read input data (text) from input file / find parameter max_char_in_word, max_words_in_sentence
    :param fileName: filename containing input text
    :return: max_char_in_word, max_words_in_sentence
    """

    character_list = []
    max_char_in_word = 0
    max_words_in_sentence = 0

    with open(fileName, "r") as file:
        for line in file:
            t_max_char = max([len(w) for w in line.split()])
            max_char_in_word = max(t_max_char, max_char_in_word)
            max_words_in_sentence = max(len(line.split()), max_words_in_sentence)

            for s in line:
                if s not in character_list and s != " " and s != "" and s.isdigit() == False:
                    character_list.append(s)
                
    return character_list, max_char_in_word, max_words_in_sentence

class CreateMatrixFromSentence(object):
    '''
     create input data in a matrix form from input text
     input: sent list storing a tuple of (sent1, sent2), label list , character list and parameter c, m
     output: matrix of size d*(c'*m) and label where d is fixed size of characters, c' = max word length (plus \t splitting words) and m = max sentence length
     '''

    def __init__(self,character_dict,c,m):
        self.__character_dict = character_dict
        self.__c = c
        self.__m = m

    def createMatrix(self,sent):
        v = len(self.__character_dict)
        mat = np.zeros((v,self.__c*self.__m),dtype=np.float32)
        index_tab = self.__character_dict["\t"]

        ss = sent.split()

        # pad from end of sentence to the end of the array
        mat[index_tab,len(ss)*self.__c:] = 1

        # process the words
        for i,word in enumerate(ss):
            # split the word by character
            ch = list(ss[i])
            ch = [a for a in ch if a.isdigit() == False]
            space = self.__c - len(ch)
            # add equal # of \t before and after word
            if space % 2 == 0:
                block_ch = ["\t"] * int(space / 2) + ch + ["\t"] * int(space / 2)
            # add more \t after word
            else:
                block_ch = ["\t"] * int(space / 2) + ch + ["\t"] * int((space / 2) + 1)
            for j,char in enumerate(block_ch):
                mat[self.__character_dict[char],i*self.__c+j] = 1

        return mat

class ComputeDistance(object):
    """
     compute distance between 2 sentences (sen1 and sen2)
     input: folder name that contain training data for word2vec, word_dist = measure to compute distance between words (i.e. "word2vec," "lev"), sen1 and sen2
     output: distance between sen1 and sen2
     """

    def __init__(self,word_dist,folder):
        self.__word_dist = word_dist
        self.__folder = folder

    def computeDistance(self, sen1, sen2):

        if self.__word_dist == "word2vec":
            s = dtw.MySentences(self.__folder)
            model = Word2Vec(s, size=8, min_count=0)
            model.save('train_model')

            # load model
            model = Word2Vec.load('train_model')

            dist = dtw.DTW_dist(sen1, sen2, model, normalize="TRUE", word_dist="word2vec")

        elif self.__word_dist == "lev":
            model = "NA"
            dist = dtw.DTW_dist(sen1, sen2, model, normalize="TRUE", word_dist=self.__word_dist)

        return dist
