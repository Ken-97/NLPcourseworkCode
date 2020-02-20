from os.path import exists

with open("./train.enzh.src", "r") as enzh_src:
  print("Source: ",enzh_src.readline())
with open("./train.enzh.mt", "r") as enzh_mt:
  print("Translation: ",enzh_mt.readline())
with open("./train.enzh.scores", "r") as enzh_scores:
  print("Score: ",enzh_scores.readline())

##################################
# English embedding with glove
##################################
import torchtext
import spacy

#Embeddings
glove = torchtext.vocab.GloVe(name='6B', dim=100)

#tokenizer model
nlp_en =spacy.load('en300')

#ENGLISH EMBEDDINGS methods from the section GERMAN-ENGLISH
# The difference from previous section is that we will use Glove embeddings directly because we are using a smaller model that spacy doesn't have
# We add a method to compute the word embedding and a method to compute the sentence embedding by averaging the word vectors

import numpy as np
import torch
from nltk import download
from nltk.corpus import stopwords

#downloading stopwords from the nltk package
download('stopwords') #stopwords dictionary, run once
stop_words_en = set(stopwords.words('english'))

def preprocess(sentence,nlp):
    text = sentence.lower()
    doc = [token.lemma_ for token in  nlp.tokenizer(text)]
    doc = [word for word in doc if word not in stop_words_en]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def get_word_vector(embeddings, word):
    try:
      vec = embeddings.vectors[embeddings.stoi[word]]
      return vec
    except KeyError:
      #print(f"Word {word} does not exist")
      pass

def get_sentence_vector(embeddings,line):
  vectors = []
  for w in line:
    emb = get_word_vector(embeddings,w)
    #do not add if the word is out of vocabulary
    if emb is not None:
      vectors.append(emb)
   
  return torch.mean(torch.stack(vectors))


def get_embeddings(f,embeddings,lang):
  file = open(f) 
  lines = file.readlines() 
  sentences_vectors =[]

  for l in lines:
    sentence= preprocess(l,lang)
    try:
      vec = get_sentence_vector(embeddings,sentence)
      sentences_vectors.append(vec)
    except:
      sentences_vectors.append(0)

  return sentences_vectors

#########################################
# LODADING CHINESE WORD2VEC EMBEDDINGS. #
#########################################

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

wv_from_bin = KeyedVectors.load_word2vec_format("model.bin", binary=True)


##########################
# PRE-PROCESSING CHINESE #
##########################
import string
import jieba
import gensim 
import spacy
import numpy as np

stop_words = [ line.rstrip() for line in open('./stopwords.dat',"r", encoding="utf-8") ]

def processing_zh(sentence): # a chinese sentence is inputed here.
  seg_list = jieba.lcut(sentence, cut_all = True) # obtain a list containing different tokens
  doc = [word for word in seg_list if word not in stop_words and word != ' ' and word != '\n'] # erase the stop words
  docs = [e for e in doc if e.isalnum()] # only put the alphanumeric things into the documents
  return docs

def get_sentence_vector_zh(line):
  vectors = []
  for w in line:
    try:
      emb = wv_from_bin[w] # obtain embedding from our embedding table with. a dimension of 100
      vectors.append(emb) # embeddings are concatenated one after another, we change the row rather than the column
    except:
      pass #Do not add if the word is out of vocabulary
  if vectors:
    vectors = np.array(vectors)
    return np.average(vectors)  # we would return a number, which represents the whole sentence.
  else:
    return 0

def get_sentence_embeddings_zh(f):
  file = open(f) 
  lines = file.readlines() 
  sentences_vectors =[]
  for l in lines:
    sent  = processing_zh(l)
    vec = get_sentence_vector_zh(sent)

    if vec is not None:
      sentences_vectors.append(vec)
    else:
      print(l)
  return sentences_vectors

import spacy
import torchtext
from torchtext import data

# two scalar obtained after 'get embeddings' methods.
zh_train_mt = get_sentence_embeddings_zh("./train.enzh.mt") # Translation Result, we would get embeddings of sentennces, one scalar for one sentence
zh_train_src = get_embeddings("./train.enzh.src",glove,nlp_en) # English source text
# print(len(zh_train_mt))
f_train_scores = open("./train.enzh.scores",'r')
zh_train_scores = f_train_scores.readlines() # Translation Score


zh_val_src = get_embeddings("./dev.enzh.src",glove,nlp_en)
zh_val_mt = get_sentence_embeddings_zh("./dev.enzh.mt")
f_val_scores = open("./dev.enzh.scores",'r')
zh_val_scores = f_val_scores.readlines()


import numpy as np


X_train= [np.array(zh_train_src),np.array(zh_train_mt)] # dimension 2x7000
X_train_zh = np.array(X_train).transpose() # dimension 7000 x 2

X_val = [np.array(zh_val_src),np.array(zh_val_mt)]
X_val_zh = np.array(X_val).transpose()

#Scores
train_scores = np.array(zh_train_scores).astype(float)
y_train_zh =train_scores

val_scores = np.array(zh_val_scores).astype(float)
y_val_zh =val_scores

######################
# TRAINING REGRESSOR #
######################
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

from sklearn.svm import SVR
from scipy.stats.stats import pearsonr

for k in ['linear','poly','rbf','sigmoid']:
    clf_t = SVR(kernel=k)
    clf_t.fit(X_train_zh, y_train_zh)
    print(k)
    predictions = clf_t.predict(X_val_zh)
    pearson = pearsonr(y_val_zh, predictions)
    # print(pearson)
    print(f'RMSE: {rmse(predictions,y_val_zh)} Pearson {pearson[0]}')
    print()