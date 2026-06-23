import numpy as np

def bag_of_words_vector(tokens, vocab):
	bow = [tokens.count(v) for v in vocab]
	bow = np.array(bow, dtype=int)
	return bow