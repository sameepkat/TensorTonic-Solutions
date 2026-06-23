def remove_stopwords(tokens, stopwords):
	return [token for token in tokens if token not in stopwords]