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

# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())
#
#
# en_train_src_sif = get_embeddings_sif("./train.enzh.src",glove,nlp_en)
# en_val_src_sif = get_embeddings_sif("./dev.enzh.src",glove,nlp_en)
# zh_train_mt_sif = get_sentence_embeddings_zh_sif("./train.enzh.mt")
# zh_val_mt_sif = get_sentence_embeddings_zh_sif("./dev.enzh.mt")
#
# X_train = np.concatenate((en_train_src_sif, zh_train_mt_sif), axis=1) # 7000 x 200
# X_val = np.concatenate((en_val_src_sif, zh_val_mt_sif), axis=1)
#
# ############################
# # SVR Regressor
# ############################
#
# from sklearn.svm import SVR
# from scipy.stats.stats import pearsonr
#
# for k in ['linear','poly','rbf','sigmoid']:
#     clf_t = SVR(kernel=k, gamma="auto")
#     clf_t.fit(X_train, train_scores)
#     print(k)
#     predictions = clf_t.predict(X_val)
#     pearson = pearsonr(val_scores, predictions)
#     print(pearson)
#     print(f'RMSE: {rmse(predictions,val_scores)} Pearson {pearson[0]}')
#     print()

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

############################
# NNdataset
############################

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np

pretrained_weights = 'distilbert-base-uncased'


class RegressionDataset:
  '''
  load english and chinese corpus via their pathes. tokenize them with process function.
  Then we create idx2word dict with class Vocabulary for all the words in the corresponding corpus.
  Using this dict, we convert all the words, start stop as well as padding sign in each sentence into vector filled with
  idx.
  Then we concatenate both english and chinese sentences matrix as our input data, we load the corresponding scores as labels.
  '''
  def __init__(self, src_file_path, tar_file_path, score_path):
    self.src_file = src_file_path
    self.tar_file = tar_file_path
    self.src_vocab = Vocabulary()
    self.src_vocab.build_from_file(src_file_path)
    self.tar_vocab = Vocabulary()
    self.tar_vocab.build_from_file(tar_file_path)
    self.score_path = score_path
    self.padded_idxs = None
    self.attention_mask = None

    sents_idx_en = []

    with open(self.src_file, "r") as f:
      lines = f.readlines()
      for l in lines:
        sentence = self.processing_zh(l)
        idxs = self.src_vocab.convert_words_to_idxs(sentence, add_eos=True)
        sents_idx_en.append(idxs)

    sents_src_transformed = [torch.tensor(sent).unsqueeze(1) for sent in sents_idx_en]
    padded_idxs_src = pad_sequence([sent for sent in sents_src_transformed]).transpose(0, 1).squeeze(2)

    sents_idx_zh = []
    with open(self.tar_file, "r") as f:
      lines = f.readlines()
      for l in lines:
        sentence = self.processing_zh(l)
        idxs = self.src_vocab.convert_words_to_idxs(sentence, add_eos=True)
        sents_idx_zh.append(idxs)

    sents_tar_transformed = [torch.tensor(sent).unsqueeze(1) for sent in sents_idx_zh]
    padded_idxs_tar = pad_sequence([sent for sent in sents_tar_transformed]).transpose(0, 1).squeeze(2)


    self.padded_idxs = torch.cat((padded_idxs_src, padded_idxs_tar), dim=1)
    zeros = torch.zeros(self.padded_idxs.shape)
    ones = torch.ones(self.padded_idxs.shape)
    self.attention_mask = torch.where(self.padded_idxs != 0, ones, zeros)

  def __getitem__(self, idx):

    score = self.get_scores()
    return self.padded_idxs[idx], score[idx], self.attention_mask

  def __len__(self):
    return len(self.padded_idxs)

  def get_scores(self):
    with open(self.score_path, "r") as f:
      scores = f.readlines()
      scores = np.array(scores).astype(float)
      scores = torch.tensor(scores)
      return scores

  @staticmethod
  def processing_zh(sentence):
    # a chinese sentence is inputed and tokenized
    seg_list = jieba.lcut(sentence, cut_all=True)
    doc = [word for word in seg_list if word not in stop_words and word != ' ' and word != '\n']
    docs = [e for e in doc if e.isalnum()]
    return docs


class Vocabulary(object):
  """Data structure representing the vocabulary of a corpus."""

  def __init__(self):
    # Mapping from tokens to integers
    self._word2idx = {}

    # Reverse-mapping from integers to tokens
    self.idx2word = []

    # 0-padding token
    self.add_word('<pad>')
    # sentence start
    self.add_word('<s>')
    # sentence end
    self.add_word('</s>')
    # Unknown words
    self.add_word('<unk>')

    self._pad_idx = self._word2idx['<pad>']
    self._bos_idx = self._word2idx['<s>']
    self._eos_idx = self._word2idx['</s>']
    self._unk_idx = self._word2idx['<unk>']

  def word2idx(self, word):
    """Returns the integer ID of the word or <unk> if not found."""
    return self._word2idx.get(word, self._unk_idx)

  def add_word(self, word):
    """Adds the `word` into the vocabulary."""
    if word not in self._word2idx:
      self.idx2word.append(word)
      self._word2idx[word] = len(self.idx2word) - 1

  def build_from_file(self, fname):
    """Builds a vocabulary from a given corpus file."""
    with open(fname) as f:
      for line in f:
        words = line.strip().split()
        for word in words:
          self.add_word(word)

  def convert_idxs_to_words(self, idxs, until_eos=False):
    """Converts a list of indices to words."""
    if until_eos:
      try:
        idxs = idxs[:idxs.index(self.word2idx('</s>'))]
      except ValueError:
        pass

    return ' '.join(self.idx2word[idx] for idx in idxs)

  def convert_words_to_idxs(self, words, add_bos=False, add_eos=False):
    """Converts a list of words to a list of indices."""
    idxs = [self.word2idx(w) for w in words]
    if add_bos:
      idxs.insert(0, self.word2idx('<s>'))
    if add_eos:
      idxs.append(self.word2idx('</s>'))
    return idxs

  def __len__(self):
    """Returns the size of the vocabulary."""
    return len(self.idx2word)


batch_size = 128


train_dat = RegressionDataset("train.enzh.src", "train.enzh.mt", "train.enzh.scores")
loader_train = DataLoader(train_dat, batch_size, shuffle=True)

samples, scores, mask = next(iter(loader_train))
print(samples.shape)

test_dat = RegressionDataset("dev.enzh.src", "dev.enzh.mt", "dev.enzh.scores")
loader_train = DataLoader(train_dat, batch_size, shuffle=True)


# tokenizer = DistilBertTokenizer()
# model = DistilBertModel()
#
# tokenizer = tokenizer.from_pretrained(pretrained_weights)
# model = model.from_pretrained(pretrained_weights)
