import numpy as np 

def euclidean_distance(x,y):
    if len(x) != len(y):
        raise ValueError("Unqueal size")
    sum = 0

    for(xi, yi) in zip(x,y):
        sum += (xi-yi) ** 2
    return sum ** 0.5