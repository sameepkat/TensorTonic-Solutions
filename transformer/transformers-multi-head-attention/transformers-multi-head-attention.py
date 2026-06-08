import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # d_model = len(Q[0][-1]) # 4
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads # 4//2 = 2
    
    Q_ = Q @ W_q # (batch_size, seq_len, d_model) = (1, 3, 4)
    K_ = K @ W_k
    V_ = V @ W_v
    
    Q__ = Q_.reshape(batch_size, -1, num_heads, d_k) # (1, 3, 2, 2)
    K__ = K_.reshape(batch_size, -1, num_heads, d_k)
    V__ = V_.reshape(batch_size, -1, num_heads, d_k)

    Q___ = Q__.transpose(0, 2, 1, 3) # (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
    K___ = K__.transpose(0, 2, 1, 3)
    K___T = K___.transpose(0, 1, 3, 2) # =>(batch_size, h, d_k, seq_len)
    V___ = V__.transpose(0, 2, 1, 3)

    score = Q___ @ K___T
    scaled = score  / (d_k ** 0.5)
    att = softmax(scaled) @ V___

    output = att.transpose(0, 2, 1, 3).reshape(batch_size, -1, d_model)
    final = output @ W_o
    return final