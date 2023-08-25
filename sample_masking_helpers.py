from scipy import ndimage as ndi
from skimage import morphology
import numpy as np
import torch
import cv2 


def thresholding(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if np.max(image) > 1: image = image.astype(np.float64) / 255
    return torch.where(torch.Tensor(image) < threshold, 0, 1).numpy().astype(np.uint8) * 255


def median_filter(inputImage: np.ndarray, ellipse_size: int = 7) -> np.ndarray:
    medBlurred = cv2.medianBlur(inputImage.astype(np.uint8), 1) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ellipse_size, ellipse_size))
    return  cv2.morphologyEx(medBlurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    # return med


def make_mask(image: np.ndarray, threshold: float, ellipse_size: int) -> np.ndarray:
    # return thresholding(image, threshold) # median_filter(thresholding(image, threshold) * 255, ksize) # thresholding(median_filter((image * 255), ksize), threshold)
    return median_filter(thresholding(image, threshold), ellipse_size) #thresholding(image, threshold) # median_filter(thresholding(image, threshold) * 255, ksize) # thresholding(median_filter((image * 255), ksize), threshold)


def remove_holes_objects(mask):
    fh = ndi.binary_fill_holes(mask)   
    m2 = morphology.remove_small_holes(fh, 500)
    return m2

