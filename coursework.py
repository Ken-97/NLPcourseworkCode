from os.path import exists

##################################
# English embedding with glove
##################################
import torchtext
import spacy
import random
import numpy as np
from sklearn.decomposition import PCA
from torchtext import data
from sklearn.decomposition import TruncatedSVD


pca = PCA(n_components = 1)

#Embeddings
glove = torchtext.vocab.GloVe(name='6B', dim=100)

#tokenizer model
spacy.load('en_core_web_sm')
nlp_en = spacy.load('en300')

#ENGLISH EMBEDDINGS methods from the section GERMAN-ENGLISH
# The difference from previous section is that we will use Glove embeddings directly because we are using a smaller model that spacy doesn't have
# We add a method to compute the word embedding and a method to compute the sentence embedding by averaging the word vectors

import numpy as np
import torch
from nltk import download
from nltk.corpus import stopwords

#downloading stopwords from the nltk package
# download('stopwords') #stopwords dictionary, run once
stop_words_en = set(stopwords.words('english'))


def preprocess(sentence:str, nlp):

    text = sentence.lower()
    doc = [token.lemma_ for token in nlp.tokenizer(text)]
    doc = [word for word in doc if word not in stop_words_en]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def get_word_vector(embeddings, word):
  # get word embedding from glove

    try:
      vec = embeddings.vectors[embeddings.stoi[word]]
      return vec
    except KeyError:
      zeros = [1 for n in range(100)]
      return zeros

def get_sentence_vector(embeddings,line):
  vectors = []
  for w in line:
    emb = get_word_vector(embeddings,w)
    vectors.append(emb) # if the word is out of the vocabulary, we would add 0s into 'vectors'
  vectors = np.array(vectors) # if we have 5 words, vectors would be 5 x 100
  one_word = False
  if len(one_word) < 2:
    one_word = True
  vectors = vectors.transpose() # now, the dimension is 100 x 5
  if not one_word:
    vectors = pca.fit_transform(vectors) # now the dimension is 100 x 1
  vectors = vectors.reshape(1, 100)
  return vectors
  # return torch.mean(torch.stack(vectors))


def get_embeddings(f,embeddings,lang):
  file = open(f) 
  lines = file.readlines() 
  sentences_vectors =[]

  for l in lines:
    sentence= preprocess(l,lang)
    try:
      vec = get_sentence_vector(embeddings,sentence)
      sentences_vectors.extend(vec)
    except:
      zeros = np.ones((1,100))
      sentences_vectors.extend(zeros)
  
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

# get stop words from file
stop_words = [ line.rstrip() for line in open('./stopwords.dat',"r", encoding="utf-8") ]


def processing_zh(sentence):
  # a chinese sentence is inputed and tokenized
  seg_list = jieba.lcut(sentence, cut_all = True) # obtain a list containing different tokens
  doc = [word for word in seg_list if word not in stop_words and word != ' ' and word != '\n'] # erase the stop words
  docs = [e for e in doc if e.isalnum()] # only put the alphanumeric things into the documents
  return docs

def get_sentence_vector_zh(line):
  '''
  Convert the line/sentence into a 100-dimension vector

  Concatenate the embeddings of the words in a sentence with wv_from_bin. Reduce the features dimension using pca.
  :param line: a Chinese sentence
  :return:
  '''
  vectors = []
  for w in line:
    try:
      emb = wv_from_bin[w] # obtain embedding from our embedding table with. a dimension of 100
      vectors.append(emb) # embeddings are concatenated one after another, we change the row rather than the column
    except:
      zeros = [random.random()/10000 for n in range(100)]
      vectors.append(zeros)
    # print('PERFORM PCA ON CHINESE SENTENCES...')
  vectors = np.array(vectors) # 100
  one_word = False
  if len(vectors) < 2:
    one_word = True
  vectors = vectors.transpose() # now, the dimension is 100 x 5
  if not one_word:
    vectors = pca.fit_transform(vectors) # now the dimension is 100 x 1
  vectors = vectors.reshape(1, 100)
  return vectors
  # return np.average(vectors)  # we would return a number, which represents the whole sentence.

def get_sentence_embeddings_zh(f):

  file = open(f)
  lines = file.readlines()
  sentences_vectors =[] # 7000 x 100 in the future

  for l in lines:
    sent = processing_zh(l)
    vec = get_sentence_vector_zh(sent)
    if vec is not None:
      sentences_vectors.extend(vec)
    else:
      print(l)
  return sentences_vectors

    
#################################
# smooth inverse frequency (SIF)
################################
def calculate_words(lines,lang):
  '''
  calculate the size of vocab
  :param lines: corpas
  :param lang:  nlp_en
  :return:
  '''
  word_size = 0
  word_dic={}
  for l in lines:
    sentence= preprocess(l,lang)
    for w in sentence:
      if w not in word_dic:
        word_dic[w]=1
      else:
        word_dic[w] =word_dic[w]+1
      word_size = word_size +1
  return word_size,word_dic

def get_sentence_vector_sif(embeddings,line,word_size,dic):
  '''
  Get sentence vector for english sentence
  :param embeddings: glove
  :param line:  english sentence
  :param word_size:  len(dic)
  :param dic: dictionary of vocabulary
  :return:
  '''
  alpha = 0.3
  emb = torch.zeros(100)
  for w in line:
    pw = (dic[w]/word_size)
    word_vectors = get_word_vector(embeddings, w)
    emb = emb +(alpha/(alpha+pw))*word_vectors  # Aplly SIF method

  emb = emb/len(line)

  return np.array(emb)

def get_embeddings_sif(f,embeddings,lang):
  '''
  Convert the English file into sentences vectors
  :param f: files for english
  :param embeddings: glove
  :param lang: nlp_en
  :return: sentences_vectors sentences embedding of the whole corpus
  '''
  file = open(f) 
  lines = file.readlines() 
  sentences_vectors =[]
  word_size, dic = calculate_words(lines,lang)

  for l in lines:
    sentence= preprocess(l,lang)
    try:
      vec = get_sentence_vector_sif(embeddings,sentence,word_size,dic)
      sentences_vectors.append(vec)
    except:
      ones= np.ones((1,100))*1e-5
      sentences_vectors.extend(ones)

  X = np.array(sentences_vectors)
  sentence_elite = rm_principal_component(X)

  return sentence_elite


def get_sentence_vector_zh_sif(line, word_size, dic):
  '''
  Convert a chinese sentence into an embedding with word2vec

  We use a weighted average to approximate our MAP of p(s|c_s)
  :param line:
  :param word_size:
  :param dic:
  :return:
  '''

  alpha = 0.3
  emb = np.zeros(100)
  for w in line:
    pw = (dic[w]/word_size)
    try:
      word_vectors = wv_from_bin[w]
      emb = emb +(alpha/(alpha+pw))*word_vectors
    except:
      zeros = np.random.randn(100)*1e-4
      emb = emb + zeros

  return emb

def calculate_words_zh(lines):
  '''
  Calculate the word frequency in Chinese corpus
  :param lines:
  :return: word_size,word_dic
  '''
  word_size = 0
  word_dic={}
  for l in lines:
    sentence= processing_zh(l)
    for w in sentence:
      if w not in word_dic:
        word_dic[w]=1
      else:
        word_dic[w] = word_dic[w]+1
      word_size = word_size +1
  return word_size,word_dic


def get_sentence_embeddings_zh_sif(f) -> np.ndarray:
  '''
  Convert sentences from chinese corpus into sentences embeddings
  :param f:
  :return: sentences_vectors shape: np.ndarray shape:7000 x 100
  '''
  file = open(f) 
  lines = file.readlines() 
  word_size, dic = calculate_words_zh(lines)
  sentences_vectors =[]
  for l in lines:
    sent  = processing_zh(l)
    vec = get_sentence_vector_zh_sif(sent,word_size,dic)
    if vec is not None:
      sentences_vectors.append(vec)
    else:
      ones= np.ones((1,100))*1e-5
      sentences_vectors.append(ones)

  X = np.array(sentences_vectors)
  X_elite = rm_principal_component(X)

  return X_elite


def rm_principal_component(X):
  '''
  Remove discourse c0 which is often closely related to syntax.

  :param X: shape N*100
  :return:
  '''
  svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
  svd.fit(X)
  u = svd.components_
  X_rm = X - X.dot(u.T)*u

  return X_rm


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#
# class RegressionDataset:
#     def __init__(self, file_path, lang="en"):
#         self.file = file_path
#         self.lang = lang
#         # Create vocabularies
#         self.src_vocab = Vocabulary()
#         self.src_vocab.build_from_file(f'train.{src_lang}')
#         self.trg_vocab = Vocabulary()
#         self.trg_vocab.build_from_file(f'train.{trg_lang}')
#         self.max_len = 0
#
#         with open(self.file, "r") as f:
#           lines = f.readlines()
#           sentences_list = []
#           for l in lines:
#               if self.lang == "en":
#                 sentence = self.process(l)
#               elif self.lang == "zh":
#                 sentence = self.processing_
#
#
#     def get_sentence
#
#     @staticmethod
#     def processing_zh(sentence):
#         # a chinese sentence is inputed and tokenized
#         seg_list = jieba.lcut(sentence, cut_all = True)
#         doc = [word for word in seg_list if word not in stop_words and word != ' ' and word != '\n']
#         docs = [e for e in doc if e.isalnum()]
#         return docs
#
#     @staticmethod
#     def process(sentence, nlp):
#         text = sentence.lower()
#         doc = [token.lemma_ for token in nlp.tokenizer(text)]
#         doc = [word for word in doc if word not in stop_words_en]
#         doc = [word for word in doc if word.isalpha()]  # restricts string to alphabetic characters only
#         return doc




# two scalar obtained after 'get embeddings' methods.
# print('Perform PCA on chinese ...')
# zh_train_mt = get_sentence_embeddings_zh("./train.enzh.mt") # Translation Result, we would get embeddings of sentennces, one scalar for one sentence
# print('Perform PCA on English ...')
# zh_train_src = get_embeddings("./train.enzh.src",glove,nlp_en) # English source text

#
# zh_val_src = get_embeddings("./dev.enzh.src",glove,nlp_en)
# zh_val_mt = get_sentence_embeddings_zh("./dev.enzh.mt")



# X_train = np.concatenate((zh_train_src, zh_train_mt), axis=1) # 7000 x 200
# X_train_zh = X_train
#
# X_val = np.concatenate((zh_val_src, zh_val_mt), axis = 1)
# X_val_zh = X_val




#Scores
def get_scores(file) ->np.ndarray:
    with open(file, "r") as f:
        scores = f.readlines()
        scores = np.array(scores).astype(float)
        return scores


train_scores = get_scores("./train.enzh.scores")
val_scores = get_scores("./dev.enzh.scores")


######################
# TRAINING REGRESSOR #
######################

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


en_train_src_sif = get_embeddings_sif("./train.enzh.src",glove,nlp_en)
en_val_src_sif = get_embeddings_sif("./dev.enzh.src",glove,nlp_en)
zh_train_mt_sif = get_sentence_embeddings_zh_sif("./train.enzh.mt")
zh_val_mt_sif = get_sentence_embeddings_zh_sif("./dev.enzh.mt")

X_train = np.concatenate((en_train_src_sif, zh_train_mt_sif), axis=1) # 7000 x 200
X_val = np.concatenate((en_val_src_sif, zh_val_mt_sif), axis=1)

############################
# SVR Regressor
############################

from sklearn.svm import SVR
from scipy.stats.stats import pearsonr

for k in ['linear','poly','rbf','sigmoid']:
    clf_t = SVR(kernel=k, gamma="auto")
    clf_t.fit(X_train, train_scores)
    print(k)
    predictions = clf_t.predict(X_val)
    pearson = pearsonr(val_scores, predictions)
    print(pearson)
    print(f'RMSE: {rmse(predictions,val_scores)} Pearson {pearson[0]}')
    print()

############################
# MLPR Regrssor
############################

# from sklearn.neural_network import MLPRegressor
# for k in ['identity', 'logistic', 'tanh', 'relu']:
#   mlp = MLPRegressor(activation=k)
#   # print('input shapes are {} and {}'.format(X_train_zh.shape, y_train_zh.shape))
#   mlp.fit(X_train_zh, y_train_zh)
#   print(k)
#   predictions =  mlp.predict(X_val_zh)
#   pearson = pearsonr(y_val_zh, predictions)
#   print(f'RMSE: {rmse(predictions,y_val_zh)} Pearson {pearson[0]}')
#   print()
# Combine SIF(en) and PCA(zh) + SVM 'rbf': scores 0.3

