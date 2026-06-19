import numpy as np

def relu(x):
	x = np.array(x)
	if x.ndim == 0:
		return max(0, x)
	if x.ndim == 1:
		return np.array([xi if xi>0 else 0 for xi in x])
	return np.array([relu(lst) for lst in x])