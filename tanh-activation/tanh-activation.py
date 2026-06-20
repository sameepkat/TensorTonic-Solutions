import numpy as np

def tanh(x):
	x = np.asarray(x)

	tan = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	return tan