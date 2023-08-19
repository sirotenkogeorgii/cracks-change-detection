from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
from typing import Callable
import numpy as np
import cv2
from metrics import mean_squared_error


def compare_affine_transformation(img1: np.ndarray, img2: np.ndarray, transformed: np.ndarray) -> None:
    print(f"Before affine transformation: {mean_squared_error(img2, img1)}")
    print(f"After affine transformation: {mean_squared_error(img2, transformed)}")
    print(f"After affine transformation and crop: {mean_squared_error(crop_borders(transformed, 10), crop_borders(img2, 10))}")


def crop_borders(image: np.ndarray, pixels2crop: int) -> np.ndarray:
    return image[pixels2crop : image.shape[0] - pixels2crop, pixels2crop : image.shape[1] - pixels2crop]


def affine_transformation(
    image1: np.ndarray, 
    image2: np.ndarray,  
    detector: Callable[[np.ndarray], tuple[tuple[cv2.KeyPoint], np.ndarray]] = cv2.ORB_create, 
    matcher: Callable[..., tuple[tuple[cv2.DMatch]]] = cv2.BFMatcher, 
    knn_neighbors: int = 2,
    display_keypoints: bool = False
    ) -> np.ndarray:

    detector_extractor = detector()
    keypoints1, descriptors1 = detector_extractor.detectAndCompute(image1.astype(np.uint8), None)
    keypoints2, descriptors2 = detector_extractor.detectAndCompute(image2.astype(np.uint8), None)

    matches = matcher().knnMatch(descriptors1, descriptors2, k=knn_neighbors)
    best_matches = []
    for knn_matches in matches:
        if len(knn_matches) >= 2:
            m, n = knn_matches[:2]
            if m.distance < 0.75 * n.distance:
                best_matches.append(m)
                
    best_matches_coords1 = [keypoints1[match.queryIdx].pt for match in best_matches]
    best_matches_coords2 = [keypoints2[match.trainIdx].pt for match in best_matches]
    
    if display_keypoints:
        _display_keypoints(image1, image2, keypoints1, keypoints2, best_matches)

    transformation = _find_transformation(image1, image2, best_matches_coords1, best_matches_coords2)
    transformated_image = cv2.warpAffine(image1, transformation, (image1.shape[1], image1.shape[0]))

    return transformated_image


def _display_keypoints(
    image1: np.ndarray, 
    image2: np.ndarray, 
    keypoints1: tuple[cv2.KeyPoint],
    keypoints2: tuple[cv2.KeyPoint], 
    best_matches: list[cv2.DMatch]
    ) -> None:

    source_points = np.float32([keypoints1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

    height, width = image1.shape
    points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    target = cv2.perspectiveTransform(points, matrix)

    image2 = cv2.polylines(image2, [np.int32(target)], True, 255, 3, cv2.LINE_AA)
    image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, best_matches[:3], None)

    plt.imshow(image3, 'gray')
    plt.show()


def _find_transformation(
    image1: np.ndarray, 
    image2: np.ndarray, 
    best_matches_coords1: list[tuple[float, float]], 
    best_matches_coords2: list[tuple[float, float]]
    ) -> np.ndarray:

    min_mse = float('inf')
    best_transformation = None
    keypoints_triples_num = min(len(best_matches_coords1), len(best_matches_coords2)) // 3
    for i in range(keypoints_triples_num):
        source_triple = np.array(best_matches_coords1[i * 3:3 * (i+1)]).astype(np.float32)
        target_triple = np.array(best_matches_coords2[i * 3:3 * (i+1)]).astype(np.float32)
        current_best_transformation = cv2.getAffineTransform(source_triple, target_triple)
        transformated_image = cv2.warpAffine(image1, current_best_transformation, (image1.shape[1], image1.shape[0]))

        current_mse = mean_squared_error(image2, transformated_image)
        if min_mse > current_mse:
            best_transformation = current_best_transformation
            min_mse = current_mse   

    return best_transformation


def histogram_equalizing(image1: np.ndarray, image2: np.ndarray, grayscale: bool = False) -> tuple[np.ndarray, np.ndarray]:
    matching, before_img_ind = _histogram_equalizing(image1, image2, grayscale)
    before_after_image = (2, 1) if before_img_ind == 0 else (1, 2)

    return matching[before_after_image[0]], matching[before_after_image[1]]


def _histogram_equalizing(image1: np.ndarray, image2: np.ndarray, grayscale: bool):
    before_radiation = 0
    if np.mean(image1) > np.mean(image2):
        image1, image2 = image2, image1
        before_radiation = 1

    return _histogram_matching(image1, image2, grayscale), before_radiation


def _histogram_matching(input_image: np.ndarray, reference: np.ndarray, grayscale: bool):
    matched = match_histograms(input_image.astype(np.int64), reference.astype(np.int64), channel_axis = None if grayscale else -1)

    return input_image, reference, matched


def mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def crop_patches(image: np.ndarray, patch_size: int, stride: int = 512) -> list[np.ndarray]:    
    patches = []
    # rows
    for i in range((image.shape[0]  - patch_size) // stride + 1):
        # columns
        for j in range((image.shape[1]  - patch_size) // stride + 1):
            current_patch = image[i * (patch_size // stride): (i + 1) * patch_size, j * (patch_size // stride): (j + 1) * patch_size]
            patches.append(current_patch)
            
    return patches


def filter_patches(patches: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    if not isinstance(mask, np.ndarray): mask = np.array(mask)
    mask = mask.ravel()
    if mask.size != len(patches):
        raise Exception(f"Mask size does not match the number of patches. Mask size: {mask.size}. Number of patches {len(patches)}.")

    new_patches = [patch for i, patch in enumerate(patches) if mask[i] == 1]
    
    return new_patches
