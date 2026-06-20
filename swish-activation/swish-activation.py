import numpy as np

def sigmoid(x):
	return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def swish(x):
	x = np.array(x)

	swish = x * sigmoid(x)
	return swish