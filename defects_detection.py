#!/usr/bin/env python3
from image_processing import affine_transformation, crop_patches, filter_patches, histogram_equalizing
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
from helpers import reconstruct_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import torch
import cv2
import os

# TODO: Make images to device
# TODO: Make inference with overlapping
# TODO: Debug arguments for console

# python3 defects_detection.py --input=examples/example_data/images/10_bez_crop.jpg --ref=examples/example_data/images/10_1_crop.jpg --diff --with_corners --grayscale
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Images path.", required=True)
parser.add_argument("--ref", type=str, help="Masks path.", required=True)
parser.add_argument("--target_dir", default="./defects_output", type=str, help="Directory to save the results.")
parser.add_argument("--model_path", default="./weights/segmentation/model_weights_60_norm.pth", type=str, help="Model path.")
parser.add_argument("--grayscale", action="store_true", help="Consider the image as grayscale.")
parser.add_argument("--by_patches", action="store_true", help="Return patches.")
parser.add_argument("--diff", action="store_true", help="Perform difference of predictions.")
parser.add_argument("--with_corners", action="store_true", help="Process corners of images.")
parser.add_argument("--colored", action="store_true", help="Color result.")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(42)
    cpu = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else cpu)

    input1 = cv2.imread(args.input, flags = cv2.IMREAD_GRAYSCALE if  args.grayscale else cv2.IMREAD_UNCHANGED)
    input2 = cv2.imread(args.ref, flags = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_UNCHANGED)

    input1, input2 = histogram_equalizing(input1, input2, grayscale = args.grayscale == True)
    resized1 = cv2.resize(input1, (512 * 5, 512 * 5), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(input2, (512 * 5, 512 * 5), interpolation=cv2.INTER_AREA)
    resized1 = affine_transformation(resized1, resized2, knn_neighbors=10, display_keypoints=False)

    mask = np.ones((5,5))
    if not args.with_corners:
        for i in [0, 4]: mask[i, 0], mask[i, 4] = 0, 0

    patches1 = filter_patches(crop_patches(resized1, 512), mask) 
    patches2 = filter_patches(crop_patches(resized2, 512), mask) 
    equalized_patch_pairs = [histogram_equalizing(patch1, patch2, grayscale = args.grayscale == True) for patch1, patch2 in zip(patches1, patches2)]

    model = CDUnet(out_channels=1).to(device)
    model = torch.load(args.model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    result_patches = []
    if args.diff: 
        images_preds1 = [model(image[0][None, None, ...])[0][0].detach().numpy() for image in equalized_patch_pairs] 
        images_preds2 = [model(image[1][None, None, ...])[0][0].detach().numpy() for image in equalized_patch_pairs] 
        result_patches = [abs(img1 - img2) for img1, img2 in zip(images_preds1, images_preds2)]

    else:
        for patch1, patch2 in zip(equalized_patch_pairs):
            _ = model(patch1)
            result_patches.append(model(patch2, second_prop=True).detach().numpy())

    os.makedirs(args.target_dir, exist_ok=True)

    if not args.by_patches:
        if not args.with_corners:
            result_patches.insert(0, np.zeros((512, 512)))
            result_patches.insert(4, np.zeros((512, 512)))
            result_patches.insert(20, np.zeros((512, 512)))
            result_patches.insert(24, np.zeros((512, 512)))
        reconstructed_image = reconstruct_image(result_patches)
        if args.colored: reconstructed_image = plt.get_cmap('viridis')(reconstructed_image)[:, :, :3]
        Image.fromarray((reconstructed_image * 255).astype(np.uint8)).save(f"{args.target_dir}/change_detection.png")      

    else:
        for i, result_patch in result_patches:
            if args.colored: result_patch = plt.get_cmap('viridis')(result_patch)[:, :, :3]
            Image.fromarray((result_patch * 255).astype(np.uint8)).save(f"{args.target_dir}/{i}.png")      



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)




