import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence


class RegressionDataset:
    def __init__(self, src_file_path, tar_file_path, lang="en"):
        self.src_file = src_file_path
        self.tar_file = tar_file_path
        self.lang = lang
        self.src_vocab = Vocabulary()
        self.src_vocab.build_from_file(src_file_path)
        self.tar_vocab = Vocabulary()
        self.tar_vocab.build_from_file(tar_file_path)
        self.padded_idxs = None

        sents_idx = []

        with open(self.src_file, "r") as f:
            lines = f.readlines()
            for l in lines:
                sentence = self.processing_zh(l)
                idxs = self.src_vocab.convert_words_to_idxs(sentence, add_eos=True)
                sents_idx.append(idxs)

        sents_src_transformed = [torch.Tensor(sent) for sent in sents_idx]
        padded_idxs_src = pad_sequence([*sents_src_transformed]).squeeze

        with open(self.tar_file, "r") as f:
            lines = f.readlines()
            for l in lines:
                sentence = self.processing_zh(l)
                idxs = self.src_vocab.convert_words_to_idxs(sentence, add_eos=True)
                sents_idx.append(idxs)

        sents_tar_transformed = [torch.Tensor(sent) for sent in sents_idx]
        padded_idxs_tar = pad_sequence([*sents_tar_transformed]).squeeze

        self.padded_idxs = torch.cat((padded_idxs_src, padded_idxs_tar))

    def __getitem__(self, idx):
        #TODO put in score here
        return self.padded_idxs[idx], score

    def __len__(self):
        return len(self.padded_idxs)


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

  def __repr__(self):
    return "Vocabulary with {} items".format(self.__len__())
