#!/usr/bin/env python3
import os
import cv2
import torch
import random
import argparse
import numpy as np
from metrics import IoUMetric
from helpers import init_logger
from tqdm.auto import tqdm as tq
from image_processing import crop_patches
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
from loss_funstions import OHEM, dice_bce_loss, dice_loss

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="model.pth", type=str, help="Path to save the trained model.")
parser.add_argument("--data_path", default="/kaggle/input/concrete-cells-s/concrete_cells_s/annotated/images", type=str, help="Path to the data directory.")
parser.add_argument("--labels_path", default="/kaggle/input/concrete-cells-s/concrete_cells_s/annotated/label/aggregate", type=str, help="Path to the labels directory.")
parser.add_argument("--logs_path", default="./train.log", type=str, help="Path to the save logs.")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone.")
parser.add_argument("--normalize", action="store_true", help="Normalize data.")
parser.add_argument("--image_augs", action="store_true", help="Augment images.")
parser.add_argument("--overlap", action="store_true", help="Crop patches with overlapping.")
parser.add_argument("--augmentations", action="store_true", help="Apply augmentations of masks and images.")
parser.add_argument("--ohem", action="store_true", help="Use online hard example mining.")
parser.add_argument("--loss", default="bce", choices=["bce", "dice", "combined"], help="Loss type.")
parser.add_argument("--test_proportion", default=0.1, type=float, help="Proportion of the test data.")
parser.add_argument("--epochs", default="frozen:10:1e-3,finetune:20:1e-4", type=str, help="Training epochs.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


class SegmentationDataset(Dataset):
    def __init__(
        self, 
        images, 
        masks, 
        transforms=None, 
        image_transform: bool = False, 
        probability: float = 0.5, 
        normalize: bool = False
    ) -> None:
        self.data = images
        self.masks = masks
        self.transforms = transforms
        self.image_transform = image_transform
        self.probability = probability
        self.normalize = normalize
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        mask = self.masks[index]

        if self.transforms is not None:
            image, mask = self.transforms(image, mask, self.probability, self.image_transform)
        
        image, mask = TF.to_tensor(image), TF.to_tensor(mask)
        if self.normalize: image = TF.normalize(image, mean=(0.485), std=(0.229))
            
        return image, mask


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    LOGGER = init_logger(args.logs_path)
    args.epochs = [(mode, int(epochs), float(lr)) for epoch in args.epochs.split(",") for mode, epochs, lr in [epoch.split(":")]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(epochs, model, scheduler, optimizer, criterion, train_loader, test_dataset, metric, device, init_epoch) -> None:
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            iou_score = 0.0

            batches_num = 0
            bar = tq(train_loader, postfix={"train_loss":0.0})
            for data, target in bar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)
                iou_score += metric(output, target)
                batches_num += 1
                bar.set_postfix(ordered_dict={"train_loss":loss.item()})
            if scheduler is not None: scheduler.step()
            message = "Epoch: {0:}. Loss: {1:.2f}. Train IoU: {2:.2f}. Test IoU: {3:.2f}".format(init_epoch + epoch + 1, train_loss, iou_score / batches_num, evaluate(model, test_dataset, metric, device))
            LOGGER.info(message)

    def augment_image(image: torch.Tensor, probability: float = 0.5) -> torch.Tensor:
        if random.random() <= probability: image = TF.adjust_contrast(image, random.uniform(-2, 2))
        if random.random() <= probability: image = TF.adjust_brightness(image, random.uniform(0.3, 1))  
        if random.random() <= probability: image = TF.adjust_sharpness(image, random.randint(2, 10))  
        if random.random() <= probability: image = TF.adjust_hue(image, random.uniform(-0.5, 0.5))    
        if random.random() <= probability: image = TF.gaussian_blur(image, [3,5,7,9][random.randint(0, 3)]) 
        return image

    def augment_image_mask(
        image: torch.Tensor,
        mask: torch.Tensor, 
        probability: float = 0.5,
        image_transform: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = TF.to_pil_image(image.astype(np.uint8)), TF.to_pil_image(mask.astype(np.uint8))
       
        if random.random() <= probability:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        if random.random() <= probability:
            image = TF.vflip(image)
            mask = TF.vflip(mask)  
        if random.random() <= probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)   
        if image_transform: image = augment_image(image)
        return image, mask

    def get_image(filepath: str, image_ind: int | str) -> np.ndarray:
        filepath = f"{filepath}/{image_ind}"
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    data = []
    labels = []
    images = sorted(os.listdir(args.data_path))
    masks = sorted(os.listdir(args.labels_path))
    for image_ind, mask_ind in zip(images, masks):
        data.extend(crop_patches(get_image(args.data_path, image_ind), 512, 256 if args.overlap else 512))
        labels.extend(crop_patches(get_image(args.labels_path, mask_ind), 512, 256 if args.overlap else 512))

    def evaluate(model, dataset, metric, device):
        return torch.mean(torch.Tensor([metric(model(image.to(device)[None, ...]), target.to(device)) for image, target in dataset]))

    test_size = int(len(data) * args.test_proportion)
    train_images, test_images = data[:-test_size], data[-test_size:]
    train_labels, test_labels = labels[:-test_size], labels[-test_size:]
    train_dataset = SegmentationDataset(train_images, train_labels, augment_image_mask if args.augmentations else None, image_transform=args.image_augs, normalize=args.normalize)
    test_dataset = SegmentationDataset(test_images, test_labels, None, image_transform=False, normalize=args.normalize)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = None
    if args.loss == "bce": criterion = torch.nn.functional.binary_cross_entropy
    elif args.loss == "dice": criterion = dice_loss
    elif args.loss == "combined": criterion = dice_bce_loss
    else: raise ValueError("Unsupported loss '{}'".format(args.loss))

    if args.ohem: criterion = OHEM(criterion, 0.7)

    epochs = 0
    model = CDUnet(out_channels=1, pretrained=args.pretrained).to(device)
    for mode, stage_epochs, stage_lr in args.epochs:
        if mode.startswith("frozen"): model.freeze_backbone()
        else: model.unfreeze_backbone()
        optimizer = torch.optim.NAdam(model.parameters(), lr=stage_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, stage_epochs * len(loader))
        fit(stage_epochs, model, scheduler, optimizer, criterion, loader, test_dataset, IoUMetric, device, epochs)
        epochs += stage_epochs

    torch.save(model, args.model_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)