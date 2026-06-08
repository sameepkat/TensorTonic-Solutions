import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Your code here

    embed = nn.Embedding(vocab_size, d_model)
    
    return embed

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    final_embeddings = []
    for token in tokens:
        embed = embedding(token)
        final_embeddings.append(embed * d_model ** 0.5)

    return torch.stack(final_embeddings)