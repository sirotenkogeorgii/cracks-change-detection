#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import cv2
import random
from PIL import Image
from helpers import augmentation_wrapper
import torchvision.transforms.functional as TF
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
from sample_masking_helpers import make_mask, remove_holes_objects
from ensemble import UnetEnsemble

# TODO: tune thresholds when the the models will ready
# python3 sample_masking.py --threshold=0.15

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", default="generated_images", type=str, help="Images path.")
parser.add_argument("--masks_path", default="generated_masks", type=str, help="Masks path.")
parser.add_argument("--model_path", default="weights/segmentation/model_weights_60_norm.pth", type=str, help="Model path.")
parser.add_argument("--threshold", default=0.20, type=float, help="Threshold to make mask binary.")
parser.add_argument("--ellipse_size", default=3, type=int, help="Ellipse for median filter.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # models_path = ["weights/masking/model_10.pth"]
    models_path = "weights/masking"

    augmentations = [(augmentation_wrapper(TF.vflip), True), (augmentation_wrapper(TF.hflip), True)]

    model = UnetEnsemble(models_path, device, tta_mode=True, ttas=augmentations).to(device)

    os.makedirs(args.masks_path, exist_ok=True)
    for filename in os.listdir(args.images_path):
        image = cv2.imread(f"{args.images_path}/{filename}", cv2.IMREAD_GRAYSCALE)
        model_output = model(image, normalize=True).detach().cpu().numpy()

        # mask = make_mask(model_output, args.threshold, args.ellipse_size)
        # mask = remove_holes_objects(mask)
        mask = (model_output * 255).astype(np.uint8)
        
        Image.fromarray(mask).save(f"{args.masks_path}/mask_{filename}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)