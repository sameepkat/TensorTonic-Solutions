import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,eps: float = 1e-6) -> np.ndarray:
	d_model = len(x[-1])

	mean = np.mean(x, axis=-1, keepdims=True)
	variance = np.var(x, axis=-1, keepdims=True)

	nenominator = gamma * (x-mean)
	denominator = (variance + eps ) ** 0.5
	
	LayerNorm = nenominator / denominator + beta

	return LayerNorm
    