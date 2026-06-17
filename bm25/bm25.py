import numpy as np
from collections import Counter
import math


def df(term, documents):
	return sum(1 for document in documents if term in document)

def df_list(term, document):
	return [doc.count(term) for doc in document]

def idf(term, documents, N):
	if df(term, documents) == 0:
		return 0

	n = df(term, documents)
	idf_value = math.log((N - n + 0.5)/(n + 0.5) + 1)
	return idf_value

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
	
	if len(query_tokens) == 0 or len(docs) == 0:
		return np.empty(0)
	N = len(docs) # no of documents
	D = np.array([len(doc) for doc in docs])
	avgdl = D.mean()

	scores = np.zeros(N)

	for term in query_tokens:
		IDF = idf(term, docs, N)
		f = np.array(df_list(term, docs))
		scores += IDF * (f * (k1+1)) / (f + k1 * (1-b+b*(D)/(avgdl)))

	return scores
	