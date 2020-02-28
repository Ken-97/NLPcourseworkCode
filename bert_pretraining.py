import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import numpy as np
import torch.nn as nn
import os
from scipy.stats.stats import pearsonr
import torchtext
import spacy
import jieba
from nltk.corpus import stopwords
from collections import Counter
from nltk import download

#Embeddings
glove = torchtext.vocab.GloVe(name='6B', dim=100)

#tokenizer model
nlp_en =spacy.load('en300')

download('stopwords') #stopwords dictionary, run once
stop_words = set(stopwords.words('english'))

def en_tokenizer(sentence):
  text = sentence.lower()
  doc = [token.lemma_ for token in nlp_en.tokenizer(text)]
  doc = [word for word in doc if word not in stop_words]
  doc = [word for word in doc if word.isalpha()]  # restricts string to alphabetic characters only
  return doc

def zh_tokenizer(sentence):
  # a chinese sentence is inputed and tokenized
  seg_list = jieba.lcut(sentence, cut_all=True)
  doc = [word for word in seg_list if word not in stop_words and word != ' ' and word != '\n']
  docs = [e for e in doc if e.isalnum()]
  return docs

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
    self.vocab = NLPVocabulary()
    self.vocab.add_from_file(fname=src_file_path, lang="en")
    self.vocab.add_from_file(fname=tar_file_path, lang="zh")
    self.score_path = score_path
    self.padded_idxs = None
    self.attention_mask = None

    print(len(self.vocab))

    sents_idx_en = []

    with open(self.src_file, "r") as f:
      lines = f.readlines()
      for l in lines:
        sentence = en_tokenizer(l)
        idxs = self.vocab.convert_words_to_idxs(sentence, add_eos=True)
        sents_idx_en.append(idxs)

    sents_src_transformed = [torch.tensor(sent).unsqueeze(1) for sent in sents_idx_en]
    padded_idxs_src = pad_sequence([sent for sent in sents_src_transformed]).transpose(0, 1).squeeze(2)

    sents_idx_zh = []
    with open(self.tar_file, "r") as f:
      lines = f.readlines()
      for l in lines:
        sentence = zh_tokenizer(l)
        idxs = self.vocab.convert_words_to_idxs(sentence, add_eos=True)
        sents_idx_zh.append(idxs)

    sents_tar_transformed = [torch.tensor(sent).unsqueeze(1) for sent in sents_idx_zh]
    padded_idxs_tar = pad_sequence([sent for sent in sents_tar_transformed]).transpose(0, 1).squeeze(2)

    self.padded_idxs = torch.cat((padded_idxs_src, padded_idxs_tar), dim=1)
    zeros = torch.zeros(self.padded_idxs.shape)
    ones = torch.ones(self.padded_idxs.shape)
    self.attention_mask = torch.where(self.padded_idxs != 0, ones, zeros)

  def __getitem__(self, idx):

    score = self.get_scores()
    return self.padded_idxs[idx], score[idx], self.attention_mask[idx]

  def __len__(self):
    return len(self.padded_idxs)

  def get_scores(self):
    with open(self.score_path, "r") as f:
      scores = f.readlines()
      scores = np.array(scores).astype(float)
      scores = torch.tensor(scores)
      return scores



class NLPVocabulary(object):
  '''
  Choose whether to add bos or eos to each sentence by setting add_bos add_eos
  Choose min_size to kick out less frequent vocab.
  '''

  def __init__(self):

    self._word2idx = {}
    self.vocab_counter = Counter()
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
    if word not in self._word2idx:
      idx = self._unk_idx
    else:
      idx = self._word2idx[word]
    return idx

  def add_word(self, word):
    if word not in self._word2idx:
      if not self._word2idx:
        self._word2idx[word] = 0
      else:
        self._word2idx[word] = len(self._word2idx)

  def count_word(self, word):

    self.vocab_counter[word] += 1

  def add_from_file(self, fname, lang):

    with open(fname) as f:
      if lang == "en":
        for line in f:
          tokens = en_tokenizer(line)
          for tk in tokens:
            self.count_word(tk)

      if lang == "zh":
        for line in f:
          tokens = zh_tokenizer(line)
          for tk in tokens:
            self.count_word(tk)

  def rm_less_frequent(self, min_size):
    '''
    remove vocabulary that is less frequent
    :return:
    '''
    self._list_reduced = [word for word, item in self.vocab_counter.items() if item > min_size]
    for idx, word in enumerate(self._list_reduced):
      self._word2idx[word] = idx+3

  def convert_words_to_idxs(self, words, add_bos=False, add_eos=False):

    idxs = [self.word2idx(w) for w in words]
    if add_bos:
      idxs.insert(0, self.word2idx('<s>'))
    if add_eos:
      idxs.append(self.word2idx('</s>'))
    return idxs

  def __len__(self):
    return len(self._word2idx)


class RegressionBert(nn.Module):
  '''
  Wrapped Bert pretrained model.
  '''
  def __init__(self, transformer_model):
    super(RegressionBert, self).__init__()
    self.transformer = transformer_model
    self.fc1 = nn.Linear(768, 1)

  def forward(self, input_ids=None, attention_mask=None):
    '''
    The outputs from distil_bert pretrained model are last-layer hidden-state, (all hidden_states), (all attentions)
    Since we are more interested in last hidden states, we slice them out. However, the shape of last hidden states are
    batch_size* sentence_length * 768, which is still extremely big for fc layers. For simplicity, we take the first token of
    each sentence and send them into fc layers. The outputs from fc layers are scores of each translation pairs, which is
    exactly what we want. Using mse loss, we do a backward propagation over our model.
    :param inputs: shape batch_size * sentence_length(padded)
    :param attention_mask:
    :return:
    '''
    output_tuples = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden_states = output_tuples[0]
    features = last_hidden_states[:, 0, :].squeeze(1)
    out = self.fc1(features)
    out = out.squeeze(1)
    return out


# Define hyperparameters
batch_size = 128
epoch_num = 5
lr_rate = 0.001


device_idx = 0
GPU = True
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# initialize tokenizer
tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")
train_dat = RegressionDataset("train.enzh.src", "train.enzh.mt", "train.enzh.scores")
loader_train = DataLoader(train_dat, batch_size, shuffle=True)
test_dat = RegressionDataset("dev.enzh.src", "dev.enzh.mt", "dev.enzh.scores")
loader_test = DataLoader(test_dat, batch_size, shuffle=True)


bert = BertModel.from_pretrained("bert-base-uncased")
model = RegressionBert(transformer_model=bert)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
criterion = nn.MSELoss()

model.to(device)
model.train()

############################
###training
############################
for epoch in range(epoch_num):

    epoch_loss = 0
    for i, (samples, scores, mask) in enumerate(loader_train):
        samples = samples.to(device)
        mask = mask.to(device)

        predictions = model(input_ids=samples, attention_mask=mask)
        loss = criterion(scores, predictions)
        loss = loss/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"iteration:{i}, epcoh_loss:{epoch_loss}")

    print('epoch [{}/{}], loss = {:.6f}'.format(epoch, epoch_num, epoch_loss / len(loader_train)))
    print('model saving...')
    torch.save(model.state_dict(), './model.pth')

os.chdir('./')


def test_bert(
         test_en_file_path="dev.enzh.src",
         test_zh_file_path="dev.enzh.en",
         score_file_path="dev.enzh.scores",
         test_batch_size=1000):

    test_dat = RegressionDataset(test_en_file_path, test_zh_file_path, score_file_path)
    loader_test = DataLoader(test_dat, test_batch_size, shuffle=True)

    bert = BertModel.from_pretrained("bert-base-uncased")
    model = RegressionBert(transformer_model=bert)
    model = model.to(device)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
    mse = nn.MSELoss()



    test_samples, test_scores, test_attention_mask = next(iter(loader_test))

    with torch.no_grad():
        predictions = model(test_samples, test_attention_mask)
        pearson = pearsonr(test_scores, predictions)
        rmse = torch.sqrt(mse(predictions, test_scores))

    return pearson[0], rmse


pearson, rmse = test_bert()

