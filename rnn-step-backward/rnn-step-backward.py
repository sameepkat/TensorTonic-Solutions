import numpy as np

def rnn_step_backward(dh, cache):
	dh = np.asarray(dh) # dL/ dh
	x_t = np.asarray(cache[0])
	h_prev = np.asarray(cache[1])
	h_t = np.asarray(cache[2])
	W = np.asarray(cache[3])
	U = np.asarray(cache[4])
	b = np.asarray(cache[5])

	# h_t = tanh(z) => dz = dL/dz
	dz = dh * (1 - h_t ** 2) # DL/dz

	dW = np.outer(dz, x_t)
	dU = np.outer(dz, h_prev)
	db = dz

	dx  =  W.T @ dz
	dh_prev = U.T @ dz
	
	return dx, dh_prev, dW, dU, db