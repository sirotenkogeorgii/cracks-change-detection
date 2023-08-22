import numpy as np
import torch


def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape.")
    return np.mean((image1 - image2) ** 2)


def IoUMetric(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction_shape = list(prediction.shape)
    prediction = torch.reshape(prediction >= 0.5, [prediction_shape[0], torch.prod(torch.Tensor(prediction_shape[1:]), dtype=torch.int32)])
    
    mask_shape = list(mask.shape)
    mask = torch.reshape(mask >= 0.5, [mask_shape[0], torch.prod(torch.Tensor(mask_shape[1:]), dtype=torch.int32)])
    
    intersection_mask = torch.logical_and(prediction, mask)
    union_mask = torch.logical_or(prediction, mask)
    
    intersection = torch.sum(intersection_mask.type(torch.FloatTensor))
    union = torch.sum(union_mask.type(torch.FloatTensor))
    
    return torch.where(union == 0, 1., intersection / union)