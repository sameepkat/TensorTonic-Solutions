import numpy as np

def manhattan_distance(x, y):
	sum = 0

	for (xi,yi) in zip(x,y):
		sum += abs(xi - yi)

	return sum