import numpy as np

def L2_norm_of(x):
	return sum([item ** 2 for item in x ]) ** 0.5

def cosine_similarity(a, b):
	a = np.array(a)
	b = np.array(b)
	A = L2_norm_of(a)
	B = L2_norm_of(b)

	if A == 0 or B == 0:
		return 0
	
	similarity = np.dot(a, b) / ( A * B )

	return similarity