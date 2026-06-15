import numpy as np

def calculate_eigenvalues(matrix):
	if len(matrix) == 0:
		return None
	for row in matrix:
		if not isinstance(row, list):
			return None
	lengths = [len(r) for r in matrix]
	if  len(matrix) != len(lengths) or len(set(lengths)) != 1 or len(matrix) != lengths[0]:
		return None
	A = np.array(matrix)

	eig_vals = np.linalg.eigvals(A)
	return np.sort(eig_vals)
	