import numpy as np

def covariance_matrix(X):
	X = X - np.mean(X, axis=0)
	if len(X) < 2 or X.ndim < 2:
		return None
	return 1/(len(X)-1) * np.transpose(X) @ X 