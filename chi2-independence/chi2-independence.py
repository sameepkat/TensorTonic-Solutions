import numpy as np

def chi2_independence(C): # observed frequencies
	C = np.array(C)
	N = np.sum(C)

	row = np.array([sum(r) for r in C])
	col = sum(C[:])	

	E = np.zeros((2, 2))
	E = np.array([[ri * ci / N for ci in col] for ri in row])

	chi = np.sum([(ci-e)**2/e for ci, e in zip(C, E)])
	
	return (chi, E)