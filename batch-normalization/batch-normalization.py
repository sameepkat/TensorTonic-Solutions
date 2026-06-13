def batch_norm_forward(x, gamma, beta, eps=1e-5):
	x = np.array(x)
	gamma = np.array(gamma)
	beta = np.array(beta)
	
	N = C = H = W = 0
	d2 = True
	if x.ndim == 2:
		N, D = x.shape # N = no. of samples, D = no of features
		d2 = True
	elif x.ndim == 4:
		N, C, H, W = x.shape # N = batch size, C= no. of channels, H = height, W = width
		d2 = False
		gamma = gamma.reshape(1, C, 1, 1)
		beta = beta.reshape(1, C, 1, 1)
	
	mean = np.mean(x, axis=0) if d2 else np.mean(x, axis=(0, 2, 3), keepdims=True)
	var = np.var(x, axis=0) if d2 else np.var(x, axis=(0,2,3), keepdims=True)

	x_hat = (x - mean) / (var + eps) ** 0.5

	y = gamma * x_hat + beta

	return y