import numpy as np

def one_hot(y, num_classes=None):
	y = np.asarray(y)
	K = num_classes

	print(f"Original y: {y}")

	if K == None:
		K = np.max(y) + 1

	print(f"K is: {K}")

	hot = np.zeros((len(y), K))
	
	#y = np.asarray([[1 if idx == yi else 0 for idx, val in enumerate(y) ] for yi in y ])

	hot[np.arange(len(y)), y] = 1

	return hot