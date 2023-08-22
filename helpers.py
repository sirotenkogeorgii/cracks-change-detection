import matplotlib.pyplot as plt
import numpy as np
import random
import string
import torch


def display_image(image: np.ndarray, gray: bool = True) -> None:
    if isinstance(image, torch.Tensor) and image.requires_grad:
        image = image.detach()
    _, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray' if gray else None)
    ax.axis('off')

def augmentation_wrapper(augmentation, parameter = None):
    def augmentation_wrapped(image):
        return augmentation(image) if parameter is None else augmentation(image, parameter)
    return augmentation_wrapped

def init_logger(log_file='./train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_random_string(length: int) -> str:
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def reconstruct_image(patches: list[tuple[np.ndarray, np.ndarray]], 
    image_size: int = 512, 
    num_rows: int = 5,
    num_cols: int = 5
) -> np.ndarray:
    image_size = 512

    # combined_image = np.zeros((num_rows * image_size, num_cols * image_size), dtype=np.uint8)
    combined_image = np.zeros((num_rows * image_size, num_cols * image_size), dtype=np.float64)

    current_image_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            img = patches[current_image_index]
            combined_image[row * image_size: (row + 1) * image_size, col * image_size: (col + 1) * image_size] = img
            current_image_index += 1

    return combined_image