# 490 NLP Coursework 1

This coursework is contributed by Zhiquan Li, Bohua Peng and Yifan Zhou. Two files are provided in this repository: coursework.py and bret.py. The first python file provides the implementation of our improved baseline architecture. The second python file provides the implementation of the BERT architecture we mentioned in our report.

## Abstract

Our implementations aims to develop a regression modelfor prediction on the quality of machine trans-lation.  In this repository, we conduct several ex-periments on pre-processing, sentence embed-ding and regression models, based on the train-ing  data:  language  pairs  (English  -  Chinese) and  translation  scores. We  also  try  to  use BERT  to  train  a  regression  model  for  betterprediction. The best result comes from sim-plified SIF model along with SVM regressorwith the kernel value of ’RBF’.

## Results

We calculate the pearson value of each model on test or validation set.

|            Model            | Test Set | Validation Set |
| :-------------------------: | :------: | :------------: |
|      SIF+SVM(rbf)+POS       |   0.29   |      0.26      |
|        SIF+SVM(rbf)         |    /     |      0.27      |
| **SIF(Modified)+ SVM(rbf)** | **0.30** |    **0.29**    |
|      PCA+SVM(logistic)      |   0.25   |      0.21      |
|            BERT             |    /     |      0.06      |

From table  above, we notice that our simplified SIF model performs the best results along with SVM regressor with the kernel value of 'RBF'.  

## Module Explanation

The file coursework.py contains 5 modules that indicate our contribution to this improved baseline model.

* PCA model

  * get_embeddings_pca()
    	Input: file path for English sentences
      	Output: Sentence embedding using PCA model

  * get_sentence_embeddings_zh_pca()
    	Input: file path for Chinese sentences
      	Output: Sentence embedding using PCA model

    An example code to get a sentence embedding is provided below. Functions listed below use similar methods to get sentence embeddings.

    ```python
    zh_train_mt = get_sentence_embeddings_zh_pca("./train.enzh.mt") # Chinese
    zh_train_src = get_embeddings_pca("./train.enzh.src",glove,nlp_en) # English
    ```

* SIF model

  * get_embeddings_sif()
    	Input: file path for English sentences
      	Output: Sentence embedding using SIF model
      	
    get_sentence_embeddings_zh_sif()
     	Input: file path for Chinese sentences
    	Output: Sentence embedding using SIF model

* SIF simplified model

  * get_embeddings_sif_sip()
    	Input: file path for English sentences
      	Output: Sentence embedding using SIF model
      	
    get_sentence_embeddings_zh_sif_sip()
     	Input: file path for Chinese sentences
    	Output: Sentence embedding using SIF model

* POS extension

  * get_postagger()
    	Input: words sequence
      	Output: pos tag sequence

As for the bert.py, the key module is the model module.

* Bert end-to-end regression model
  * model(transformer_model=bert)
        Input: indices of sentence
    	Output: score tensor 

## How to test

Here we provide sample test case to help you test our code.

For the improved baseline model:

```python
zh_train_mt = get_sentence_embeddings_zh("./train.enzh.mt") 
# return the sentence embedding of Chinese sentences

zh_train_src = get_embeddings("./train.enzh.src",glove,nlp_en) # return the sentence embedding of source sentences (English)

f_train_scores = open("./train.enzh.scores",'r') # Ground Truth File

zh_train_scores = f_train_scores.readlines() # Translation Score (Ground Truth)

X_train= [np.array(zh_train_src),np.array(zh_train_mt)] # Concatenate the embeddings to form the input

X_train_zh = np.array(X_train).transpose() 

# SVR is used to perform the regression task
clf_zh = SVR(kernel='rbf')
clf_zh.fit(X_train_zh, y_train_zh)
```

For the BERT model:

```python
test_bert(
         test_en_file_path="dev.enzh.src",
         test_zh_file_path="dev.enzh.en",
         score_file_path="dev.enzh.scores",
         test_batch_size=1000)
# test_en_file_path: path of English test file
# test_zh_file_path: path of Chinese test file
# score_file_path: path of score file path
# test_batch_size: batch size of the test Dataloader
```

