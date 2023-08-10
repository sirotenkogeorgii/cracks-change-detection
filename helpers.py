import numpy as np
import random
import string

def get_random_string(length: int) -> str:
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    
    return result_str


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape.")
    return np.mean((image1 - image2) ** 2)