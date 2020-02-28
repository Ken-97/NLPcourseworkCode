This gitlab contains two files. coursework.py provides the implementation of the basic architecture in the report. bert.py provides the implementation of Bert architecure mentioned in the report.

********************** Function explanation ****************************
1)PCA model
get_embeddings_pca()
	Input: file path for English sentences
	Output: Sentence embedding using PCA model
get_sentence_embeddings_zh_pca()
	Input: file path for Chinese sentences
	Output: Sentence embedding using PCA model
	
Runing case for getting setence embedding is provided below.
zh_train_mt = get_sentence_embeddings_zh_pca("./train.enzh.mt") # Chinese
zh_train_src = get_embeddings_pca("./train.enzh.src",glove,nlp_en) # English


2)SIF model
get_embeddings_sif()
	Input: file path for English sentences
	Output: Sentence embedding using SIF model
	
get_sentence_embeddings_zh_sif()
 	Input: file path for Chinese sentences
	Output: Sentence embedding using SIF model
 
3)SIF simplified model 
get_embeddings_sif_sip()
	Input: file path for English sentences
	Output: Sentence embedding using SIF model
	
get_sentence_embeddings_zh_sif_sip()
 	Input: file path for Chinese sentences
	Output: Sentence embedding using SIF model
	
4)Pos extension
get_postagger()
	Input: words sequence
	Output: pos tag sequence

********************** Test case **********************
zh_train_mt = get_sentence_embeddings_zh("./train.enzh.mt") # Chinese

zh_train_src = get_embeddings("./train.enzh.src",glove,nlp_en) # English

f_train_scores = open("./train.enzh.scores",'r')

zh_train_scores = f_train_scores.readlines() # Translation Score 

X_train= [np.array(zh_train_src),np.array(zh_train_mt)] # dimension 2x7000

X_train_zh = np.array(X_train).transpose() 

#Predict
clf_zh = SVR(kernel='rbf')
clf_zh.fit(X_train_zh, y_train_zh)
