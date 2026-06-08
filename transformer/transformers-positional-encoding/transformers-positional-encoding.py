import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    PE = np.zeros((seq_length, d_model), dtype=np.float32)
    
    for pos in range(seq_length):
        for dim_index in range(d_model):
            power = 2*(dim_index // 2) / d_model
            if dim_index%2 == 0:
                PE[pos][dim_index] = np.sin(pos / 10000 ** power)
            else:
                PE[pos][dim_index] = np.cos(pos / 10000 ** power)
        
    
    return PE