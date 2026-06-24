def max_pooling_2d(X, pool_size):

	out_h = len(X) // pool_size
	out_w = len(X[-1]) // pool_size

	grand_out = []
	for row in range(out_h):
		out = []
		for col in range(out_w):
			max_value = X[row * pool_size][col * pool_size]

			for a in range(pool_size):
				for b in range(pool_size):
					value = X[row * pool_size + a][col * pool_size + b]
					max_value = max(max_value, value)
			out.append(max_value)
		grand_out.append(out)
	return grand_out