import numpy as np


def leaky_relu(x, alpha=0.1):
	return np.array([item if item >=0 else alpha * item for item in x])