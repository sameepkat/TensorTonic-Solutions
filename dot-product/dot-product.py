import numpy as np

def dot_product(x, y):
	if len(x) != len(y):
		raise ValueError("Dimension of x and y are not the same")
	return sum([xi * yi / 1.0 for (xi, yi) in zip(x,y )])