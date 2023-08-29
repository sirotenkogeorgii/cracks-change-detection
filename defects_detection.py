#!/usr/bin/env python3
from image_processing import affine_transformation, crop_patches, histogram_equalizing, overlap_patches
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
import torchvision.transforms.functional as TF
from helpers import reconstruct_image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import torch
import cv2
import os

# python3 defects_detection.py --input=/Users/georgiisirotenko/Downloads/2_bez_popisu_crop.jpg --ref=/Users/georgiisirotenko/Downloads/2_2_crop.jpg --colored --diff
# python3 defects_detection.py --input=examples/example_data/images/10_bez_crop.jpg --ref=examples/example_data/images/10_1_crop.jpg --diff --grayscale
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Images path.", required=True)
parser.add_argument("--ref", type=str, help="Masks path.", required=True)
parser.add_argument("--target_dir", default="defects_output", type=str, help="Directory to save the results.")
parser.add_argument("--model_path", default="weights/masking/model_39.pth", type=str, help="Model path.")
parser.add_argument("--by_patches", action="store_true", help="Return patches.")
parser.add_argument("--diff", action="store_true", help="Perform difference of predictions.")
parser.add_argument("--colored", action="store_true", help="Color result.")
parser.add_argument("--normalize", action="store_true", help="Normalize inputs.")
parser.add_argument("--overlap", action="store_true", help="Perform prediction with overlapping.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input1 = cv2.imread(args.input, flags=cv2.IMREAD_GRAYSCALE)
    input2 = cv2.imread(args.ref, flags=cv2.IMREAD_GRAYSCALE)

    input1, input2 = histogram_equalizing(input1, input2, grayscale=True)
    resized1 = cv2.resize(input1, (512 * 5, 512 * 5), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(input2, (512 * 5, 512 * 5), interpolation=cv2.INTER_AREA)
    resized1 = affine_transformation(resized1, resized2, knn_neighbors=10, display_keypoints=False)

    patches1 = crop_patches(resized1, 512, stride=512 if not args.overlap else 256)
    patches2 = crop_patches(resized2, 512, stride=512 if not args.overlap else 256)
    equalized_patch_pairs = [histogram_equalizing(patch1, patch2, grayscale=True) for patch1, patch2 in zip(patches1, patches2)]

    def to_tensor_image(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        image = TF.to_tensor(image)
        if normalize: image = TF.normalize(image, mean=(0.485), std=(0.229)) 
        return image.float()
    
    def overlap_and_crop(patches: list[np.ndarray]) -> np.ndarray:
        overlapped_patches = overlap_patches([2560, 2560], 256, patches)
        return crop_patches(overlapped_patches, 512, 512)

    model = torch.load(args.model_path, map_location=device).to(device)
    # model.eval() # NOTE: eval mode changes the result a lot. (maybe due to the batchnorm)
    model.train(True)

    result_patches = []
    if args.diff:       
        images_preds1 = [model(to_tensor_image(image[0], normalize=args.normalize)[None, ...].to(device))[0][0].detach().numpy() for image in equalized_patch_pairs] 
        images_preds2 = [model(to_tensor_image(image[1], normalize=args.normalize)[None, ...].to(device))[0][0].detach().numpy() for image in equalized_patch_pairs] 
        if args.overlap:
            images_preds1 = overlap_and_crop(images_preds1)
            images_preds2 = overlap_and_crop(images_preds2)
        result_patches = [abs(img1 - img2) for img1, img2 in zip(images_preds1, images_preds2)]

    else:
        for patch1, patch2 in equalized_patch_pairs:
            _ = model(to_tensor_image(patch1, normalize=args.normalize)[None, ...].to(device))
            result_patches.append(model(to_tensor_image(patch2, normalize=args.normalize)[None, ...].to(device), second_prop=True)[0][0].detach().numpy())
        if args.overlap: result_patches = overlap_and_crop(result_patches)

    os.makedirs(args.target_dir, exist_ok=True)
    if not args.by_patches:
        reconstructed_image = reconstruct_image(result_patches)
        if args.colored: reconstructed_image = plt.get_cmap('viridis')(reconstructed_image)[:, :, :3]
        Image.fromarray((reconstructed_image * 255).astype(np.uint8)).save(f"{args.target_dir}/change_detection.png")      

    else:
        for i, result_patch in enumerate(result_patches):
            if args.colored: result_patch = plt.get_cmap('viridis')(result_patch)[:, :, :3]
            Image.fromarray((result_patch * 255).astype(np.uint8)).save(f"{args.target_dir}/{i}.png")      


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
