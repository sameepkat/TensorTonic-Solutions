def softmax(x, axis=-1):
	e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
	return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:

	mean = np.mean(x, axis=-1, keepdims=True)
	variance = np.var(x, axis=-1, keepdims=True)

	return gamma * (x - mean) / ((variance + eps) ** 0.5 ) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray, nums_head: int) -> np.ndarray:
	batch_size, seq_len, d_model = Q.shape
	h = nums_head
	d_v = d_model // h
	
	Q_ = Q @ W_q # (batch_size, seq_len, d_model)
	K_ = K @ W_k
	V_ = V @ W_v

	Q__ = Q_.reshape(batch_size, -1, h, d_v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_v)
	K__ = K_.reshape(batch_size, -1, h, d_v)
	V__ = V_.reshape(batch_size, -1, h, d_v)
	
	Q___ = Q__.transpose(0,2, 1, 3) # (batch_size, seq_len, h, d_v) -> (batch_size, h, seq_len, d_v)
	K___ = K__.transpose(0,2, 1, 3)
	V___ = V__.transpose(0,2, 1, 3)
	K___T = K___.transpose(0, 1, 3, 2) # (batch_size, h, seq_len, d_v) -> (batch_size, h, d_v, seq_len)

	scaled_dot_product_attention = softmax(Q___ @ K___T / d_v ** 0.5) @ V___

	output = scaled_dot_product_attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, d_model)
	final_output = output @ W_o

	return final_output
	

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
	ReLU = lambda eq: np.maximum(0, eq)

	return ReLU(x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
	
	# Sub-layer 1 :Attention + Add & Norm	
	attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
	residual_attention = x + attn_out
	x_1 = layer_norm(residual_attention, gamma1, beta1) # Layer Normalization 1 

	# Sub-layer 2: FFN + Add and Norm
	ffn = feed_forward(x_1, W1, b1, W2, b2)
	residual_attention = x_1 + ffn
	x_2 = layer_norm(residual_attention, gamma2, beta2)
	
	return x_2