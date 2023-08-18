from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import torch
import cv2 


def thresholding(image, threshold: float = 0.5) -> np.ndarray:
    if np.max(image) > 1: image = image.astype(np.float64) / 255
    return torch.where(torch.Tensor(image) < threshold, 0, 1).numpy()


def MedianFilter(inputImage: np.ndarray, ksize: int = 7) -> np.ndarray:
    medBlurred = cv2.medianBlur(inputImage.astype(np.uint8), ksize) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    return  cv2.morphologyEx(medBlurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    # return med


def make_mask(image: np.ndarray, threshold: float, ksize: int) -> np.ndarray:
    return thresholding(MedianFilter((image * 255), ksize), threshold)


def remove_holes_objects(mask):
    fh = ndi.binary_fill_holes(mask)   
    #m1 = morphology.remove_small_objects(fh, 200)
    m2 = morphology.remove_small_holes(fh, 250)
    return m2

