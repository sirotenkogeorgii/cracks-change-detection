#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
from sample_masking_helpers import make_mask, remove_holes_objects

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", default="generated_images", type=str, help="Images path.")
parser.add_argument("--masks_path", default="generated_masks", type=str, help="Masks path.")
parser.add_argument("--model_path", default="weights/segmentation/model_weights_60_norm.pth", type=str, help="Model path.")
parser.add_argument("--map_location", default="cpu", type=str, help="New device for weights of the model.")
parser.add_argument("--threshold", default=0.23, type=float, help="Threshold to make mask binary.")
parser.add_argument("--kernel_size", default=1, type=int, help="Kernel size.")


if __name__ == "__main__":
    torch.manual_seed(42)
    args = parser.parse_args()

    model = CDUnet(out_channels=1)
    model = torch.load(args.model_path, map_location=args.map_location)

    os.makedirs(args.masks_path, exist_ok=True)
    for filename in os.listdir(args.images_path):
        image = cv2.imread(f"{args.images_path}/{filename}", cv2.IMREAD_GRAYSCALE)
        model_output = model(image[None, None, ...]).detach().numpy()[0][0]
        # TODO: change hp
        # mask = make_mask(model_output, args.threshold, args.kernel_size)

        mask = remove_holes_objects((model_output * 255).astype(np.uint8))
        Image.fromarray(mask).save(f"{args.masks_path}/{filename}")