import numpy as np

def ReLU(eq):
	return np.maximum(0, eq)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
				W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
	d_model = len(x[-1][-1])
	x = np.asarray(x)
	W1 = np.asarray(W1)
	b1 = np.asarray(b1)
	W2 = np.asarray(W2)
	b2 = np.asarray(b2)

	FFN = ReLU(x @ W1 + b1) @ W2 + b2

	return FFN