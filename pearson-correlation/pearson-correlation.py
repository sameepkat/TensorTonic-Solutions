import numpy as np

def pearson_correlation(X):
	X = np.array(X)
	rows = len(X)
	cols = len(X[-1])
	if rows == 0 or rows < 2 or X.ndim != 2:
		return None

	mean = np.mean(X, axis=0)

	deviation_X = X - mean
	cov_matrix = np.transpose(deviation_X) @ deviation_X / (rows - 1)
	sd = np.diag(cov_matrix) ** 0.5

	denominator = np.outer(sd, sd)
	R = cov_matrix / denominator
	return R