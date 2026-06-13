import numpy as np
from collections import Counter
import math

def tf(terms, vocab):
	counts = Counter(terms)
	total_terms = len(terms)

	x = [counts.get(term, 0) / total_terms for term in vocab]
	return x 

def df(term, tokenized_documents): # no of documents containing this term
	return sum(1 for document in tokenized_documents if term in document)

def idf(term, tokenized_documents, N): # returns idf
	
	if df(term, tokenized_documents) == 0:
		return 0
		
	idf_value = math.log(N / df(term, tokenized_documents))

	return idf_value

def tfidf(tf, idf):
	tf = np.array(tf)
	idf = np.array(idf)

	return tf * idf

def tfidf_vectorizer(documents):
	documents = np.asarray(documents)
	N = len(documents)
	
	
	tokenized_document = [sentence.lower().split() for sentence in documents]
	
	words = [word for word_list in tokenized_document for word in word_list]
	
	counts = Counter(words)
	ordered_counts = dict(counts)
	vocab = sorted(ordered_counts.keys())

	idf_vec = [idf(term, tokenized_document, N) for term in vocab]
	tfidf_matrix = []

	for document in tokenized_document:
		tf_vec = tf(document, vocab)
		tfidf_vec = tfidf(tf_vec, idf_vec)
		#print(f"TF: {tf_vec}")
		#print(f"TFIDF: {tfidf(tf_vec, idf_vec)}")
	#print(f"IDF: {idf_vec}")
		tfidf_matrix.append(tfidf_vec)	
	return np.array(tfidf_matrix), vocab