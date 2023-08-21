#!/usr/bin/env python3
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm as tq
from cd_unet import CDUnet, UpSamplingBlock, ConvBlock
import torchvision.transforms.functional as TF
from metrics import IoUMetric
import random

# TODO: Clean code
# TODO: Finish code
# TODO: Add arguments
# TODO: Add ensembling


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


def main() -> None:
    # torch.manual_seed(42)
    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    def augment_image(image: torch.Tensor, probability: float = 0.5) -> torch.Tensor:
        if random.random() <= probability: image = TF.adjust_contrast(image, random.uniform(-2, 2))
        if random.random() <= probability: image = TF.adjust_brightness(image, random.uniform(0.3, 1))  
        if random.random() <= probability: image = TF.adjust_sharpness(image, random.randint(2, 10))  
        if random.random() <= probability: image = TF.adjust_hue(image, random.uniform(-0.5, 0.5))    
        if random.random() <= probability: image = TF.gaussian_blur(image, [3,5,7,9][random.randint(0, 3)]) 
        return image

    def fit(epochs, model, scheduler, optimizer, train_loader, test_dataset, metric, device) -> None:
        model.train()
        criterion = torch.nn.BCELoss() # BCE_OHEM(0.7)
        scheduler = scheduler
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
            print("Epoch: {0:}. Loss: {1:.2f}. Train IoU: {2:.2f}. Test IoU: {3:.2f}".format(epoch + 1, train_loss, iou_score / batches_num, evaluate(model, test_dataset, metric, device)))

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

    def make512patches(image: np.ndarray, size: int) -> list[np.ndarray]:
        patches = []
        for i in range(1, (image.shape[0] // size) + 1):
            for j in range(1, (image.shape[1] // size) + 1):
                patches.append(image[size * (i - 1): size * i, size * (j - 1): size * j])
        return patches

    def get_image(filepath: str, image_ind: int | str) -> np.ndarray:
        filepath = f"{filepath}/{image_ind}"
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    DATA = "/kaggle/input/concrete-cells-s/concrete_cells_s/annotated/images"
    LABELS = "/kaggle/input/concrete-cells-s/concrete_cells_s/annotated/label/aggregate"

    images = sorted(os.listdir(DATA))
    masks = sorted(os.listdir(LABELS))
    print(len(images), len(masks))

    # NOTE: try to resize images to make them more density, and after that crop to patches
    data = []
    labels = []
    for image_ind, mask_ind in zip(images, masks):
        print(image_ind, mask_ind)
        data.extend(make512patches(get_image(DATA, image_ind), 512))
        labels.extend(make512patches(get_image(LABELS, mask_ind), 512))

    def evaluate(model, dataset, metric, device):
        return torch.mean(torch.Tensor([metric(model(image.to(device)[None, ...]), target.to(device)) for image, target in dataset]))

    train_images, test_images = data[:-25], data[-25:]
    train_labels, test_labels = labels[:-25], labels[-25:]
    # train_dataset = SegmentationDataset(train_images, train_labels, None, image_transform=False, normalize=False)
    train_dataset = SegmentationDataset(train_images, train_labels, augment_image_mask, image_transform=False, normalize=True)
    test_dataset = SegmentationDataset(test_images, test_labels, None, image_transform=False, normalize=False)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = CDUnet(out_channels=1, pretrained=True).to(device)
    init_lr = 0.0001

    freeze_epochs = 30
    unfreeze_epochs = 70


    model.freeze_backbone()
    optimizer = torch.optim.NAdam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, freeze_epochs * len(loader))
    fit(freeze_epochs, model, scheduler, optimizer, loader, test_dataset, IoUMetric, device)

    model.unfreeze_backbone()
    optimizer = torch.optim.NAdam(model.parameters(), lr=init_lr)#lr=init_lr/10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, unfreeze_epochs * len(loader))
    fit(unfreeze_epochs, model, scheduler, optimizer, loader, test_dataset, IoUMetric, device)

    torch.save(model, "/kaggle/working/model_weights_60_eff_norm.pth")

if __name__ == "__main__":
    main()