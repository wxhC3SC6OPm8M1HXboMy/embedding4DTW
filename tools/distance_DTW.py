# -*- coding: utf-8 -*-
"""
Created on Tue May 31 08:39:44 2016

@author: papis
"""

from scipy.spatial import distance
import Levenshtein as lv
import os
import re

cachedStopWords = ["a", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "after", "afterwards", "again", "against", "ah", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", "approximately", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "auth", "available", "away", "awfully", "b", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "between", "beyond", "biol", "both", "brief", "briefly", "but", "by", "c", "ca", "came", "can", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing", "contains", "could", "couldnt", "d", "date", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards", "due", "during", "e", "each", "ed", "edu", "effect", "eg", "eight", "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "et-al", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "few", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "for", "former", "formerly", "forth", "found", "four", "from", "further", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten", "h", "had", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he", "hed", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "herself", "hes", "hi", "hid", "him", "himself", "his", "hither", "home", "how", "howbeit", "however", "hundred", "i", "id", "ie", "if", "i'll", "im", "immediate", "immediately", "importance", "important", "in", "inc", "indeed", "index", "information", "instead", "into", "invention", "inward", "is", "isn't", "it", "itd", "it'll", "its", "itself", "i've", "j", "just", "k", "keep\tkeeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look", "looking", "looks", "ltd", "m", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "more", "moreover", "most", "mostly", "mr", "mrs", "much", "mug", "must", "my", "myself", "n", "na", "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "now", "nowhere", "o", "obtain", "obtained", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "omitted", "on", "once", "one", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "re", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "s", "said", "same", "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "she", "shed", "she'll", "shes", "should", "shouldn't", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure\tt", "take", "taken", "taking", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'll", "theyre", "they've", "think", "this", "those", "thou", "though", "thoughh", "thousand", "throug", "through", "throughout", "thru", "thus", "til", "tip", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "very", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "was", "wasnt", "way", "we", "wed", "welcome", "we'll", "went", "were", "werent", "we've", "what", "whatever", "what'll", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "whose", "why", "widely", "willing", "wish", "with", "within", "without", "wont", "words", "world", "would", "wouldnt", "www", "x", "y", "yes", "yet", "you", "youd", "you'll", "your", "youre", "yours", "yourself", "yourselves", "you've", "z", "zero"]

class MySentences(object):
    """
    collect input data (to train word2vec)
    """
    def __init__(self, dirname):
        self.dirname = dirname
 
            
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            infile = open(os.path.join(self.dirname, fname))
            for line in infile:
                line = line.lower()                               
                sent = line.split()
                unigram = []
                for j in sent:
                    if j not in cachedStopWords:
                        j = j.replace('"','')
                        j = re.sub(r"[^a-zA-Z\-']","",j)
                        j = ''.join([x if ord(x) < 128 else '' for x in j])
                        unigram.append(j)
#                
                unigram = [x for x in unigram if x!='' and len(x)>3]    
                yield unigram
                
            infile.close()

def path_length(lev):
    """
    Calculate the path length to normalize dtw distance
    """
    
    len1 = len(lev)-1
    len2 = len(lev[0])-1
    path=[(len1,len2)]
    i = len1
    j = len2
    while (0,0) not in path and i!=0 and j!=0:
        curr = lev[i][j]
        a = curr-lev[i-1][j-1]
        b = curr-lev[i][j-1]
        c = curr-lev[i-1][j]
        if a<0:
            a = 10000
        if b<0:
            b = 10000
        if c<0:
            c = 10000
        nextMin = min(a,b,c)
        if nextMin == a and i>=1 and j>=1:
            path.append((i-1,j-1))
            i = i-1
            j = j-1
        elif nextMin == b and i>=0 and j>=1:
            path.append((i,j-1))
            j = j-1
        elif nextMin == c and i>=1 and j>=0:
            path.append((i-1,j))
            i = i-1
    if i==0:
        while j!=0:
            j-=1
            path.append((i,j))
    if j==0:
        while i!=0:
            i-=1
            path.append((i,j))
            
    
    return len(path)-1


def _edit_dist_init_dtw(len1,len2):
    """
    Initialize the distance matrix
    """
    infinity = 1e08
    dist = []
    for i in range(len1):
        dist.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        dist[i][0] = infinity    # column 0: 0,1,2,3,4,...
    for j in range(len2):
        dist[0][j] = infinity    # row 0: 0,1,2,3,4,...
    dist[0][0] = 0
    return dist
    
def _edit_dist_step_dtw(dtw, i, j, cost):
    
    # skipping a word in sen1
    a = dtw[i - 1][j] 
    # skipping a word in sen2
    b = dtw[i][j - 1] 
    # substitution
    c = dtw[i - 1][j - 1] 

    # pick the cheapest
    dtw[i][j] = min(a, b, c)+cost

def DTW_dist(sen1,sen2,model,normalize="FALSE",word_dist="word2vec"):
#    Calculate the Dynamic Time Warping between two sentences.
#    normalize is a tag whether the computed distance is normalized with path length
#    word_dist is the method to calculate word distance (now "word2vec" vs "lev")
#    model is from word2vec (used to calculate distance between two words)
#    sen1, sen2 are sentence1 and sentence2.

    sen1 = sen1.split()
    sen2 = sen2.split()
    sen1 = [re.sub(r"[^a-zA-Z\-']","",i).lower() for i in sen1]
    sen2 = [re.sub(r"[^a-zA-Z\-']","",i).lower() for i in sen2]
    sen1 = [i for i in sen1 if i.lower() not in cachedStopWords]
    sen2 = [i for i in sen2 if i.lower() not in cachedStopWords]
    
    
    if word_dist == "word2vec":
        word = model.index2word  
        sen1 = [i for i in sen1 if i!='' and i.lower()  and len(i)>3 and i in word]
        sen2 = [i for i in sen2 if i!='' and i.lower()  and len(i)>3 and i in word]
    
    
    
    # set up a 2-D array
    len1 = len(sen1)
    len2 = len(sen2)
    dtw = _edit_dist_init_dtw(len1 + 1, len2 + 1)
    
    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            if word_dist == "word2vec":
                word_cost = 1 # default
                word1 = re.sub(r'[^a-zA-Z\-]','',sen1[i]).lower()
                word2 = re.sub(r'[^a-zA-Z\-]','',sen2[j]).lower()
                try:
                    word_cost = distance.euclidean(model[sen1[i]],model[sen2[j]])
                except KeyError:
                    # for debugging only. This error should not be displayed.
                    print("add_word_k: ",word1," ",word2)
            
            elif word_dist == "lev":
                word_cost = lv.distance(sen1[i],sen2[j])
                
            else:
                # for other word distance measures (set to default value 1 for now)
                word_cost = 1
            _edit_dist_step_dtw(dtw, i + 1, j + 1, word_cost)
    
    dist = dtw[len1][len2]
    if normalize == "TRUE":        
        # normalized distance with path length
        norm = path_length(dtw)  
        dist = float(dist)/float(norm)
        
    return dist