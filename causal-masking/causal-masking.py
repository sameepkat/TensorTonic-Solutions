import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
	scores = np.array(scores)
	new = np.zeros(scores.shape, dtype=scores.dtype)

	rows = scores.shape[-2]
	cols = scores.shape[-1]

	leading_shape = scores.shape[:-2]

	if len(leading_shape) == 0:
		for i in range(rows):
			for j in range(cols):
				if j <= i:
					new[i][j] = scores[i][j]
				else:
					new[i][j] = mask_value
	else:
		for leading_index in np.ndindex(leading_shape):
			for i in range(rows):
				for j in range(cols):
					if j <= i:
						new[leading_index][i][j] = scores[leading_index][i][j]
					else:
						new[leading_index][i][j] = mask_value


	

	return new