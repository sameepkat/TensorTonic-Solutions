import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    K_T = torch.transpose(K, 1, 2)
    d_k = len(K_T[-1])
    
    S = Q @ K_T
    S_scaled = S/ d_k ** 0.5

    attention = F.softmax(S_scaled, dim=-1) @ V
    return attention