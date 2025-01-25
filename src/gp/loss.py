import math

def hamming_distance(x: dict, y: dict) -> int:
    '''
    Calculate the Hamming distance between two vectors.
    '''
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0
    for xi, yi in zip(x, y):
        if xi != yi:
            dist += 1


def euclidean_distance(x: dict, y: dict) -> float:
    '''
    Calculate the Euclidean distance between two vectors.
    '''
    if len(x) != len(y):
        raise ValueError("Dimensions do not match.")
    dist = 0.0
    for xi, yi in zip(x, y):
        dist += math.pow(xi - yi, 2)
    return math.sqrt(dist)
