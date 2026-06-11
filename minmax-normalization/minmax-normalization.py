import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
	#x_ = X - min(X)
	minimum = np.min(X, axis, keepdims=True)
	maximum = np.max(X, axis, keepdims=True)
	arr_range = maximum - minimum

	if (arr_range == 0).any():
		normalized = (X - minimum) / (arr_range + eps)
	else:
		normalized = (X - minimum) / arr_range
	

	return normalized