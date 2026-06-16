import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
	w = np.array(w)
	g = np.array(g)
	G = np.array(G)
	
	G = G + g ** 2
	w_updated = w - lr *g / (np.sqrt(G + eps))
	return w_updated, G