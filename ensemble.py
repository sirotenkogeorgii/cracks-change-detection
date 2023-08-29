import torchvision.transforms.functional as TF
from typing import Callable
import numpy as np
import torch
import os


class UnetEnsemble(torch.nn.Module):
    def __init__(
        self, 
        models_path: list[str] | str, 
        device: torch.device, 
        tta_mode: bool = False, 
        ttas: list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]] = None
    ) -> None:
        super(UnetEnsemble, self).__init__()

        self.device = device
        self.tta_mode = tta_mode
        self.ttas = ttas
        self._models = None
        if isinstance(models_path, list):
            self._models = [torch.load(model_path, map_location=device).to(device) for model_path in models_path]
        elif isinstance(models_path, str):
            self._models = [torch.load(f"{models_path}/{model_path}", map_location=device).to(device) for model_path in os.listdir(models_path)]
        else:
            raise ValueError("Unsupported model parameter'{}'".format(models_path))

    def forward(self, inputs: torch.Tensor, normalize: bool = False) -> list[torch.Tensor]:
        if not isinstance(input, list): inputs = [inputs]
        predictions = []
        for image in inputs:
            image_prediction = None
            if self.tta_mode: image_prediction = [self._tta_inference(model, image, normalize) for model in self._models]
            else: image_prediction = [model(self._process_image(image, None, normalize)[None, ...])[0][0] for model in self._models]
            prediction = torch.mean(torch.stack(image_prediction), dim=0)
            predictions.append(prediction)

        if len(predictions) == 1: predictions = predictions[0]
        return predictions
        
    def _process_image(self, image, augmentation = None, normalize: bool = False):
        image = TF.to_pil_image(image.astype(np.uint8) if isinstance(image, np.ndarray) else image.type(torch.uint8))
        if augmentation is not None: image = augmentation(image)
        image = TF.to_tensor(image).to(self.device)
        if normalize: image = TF.normalize(image, mean=0.485, std=0.229)

        return image

    def _tta_inference(self, model, image, normalize):
        image_predictions = [model(self._process_image(image, None, normalize)[None, ...])[0][0]]
        for augmentation, invert in self.ttas:
            augmented_image = self._process_image(image, augmentation, normalize)
            image_prediction = model(augmented_image[None, ...].to(self.device))[0][0]
            if invert: image_prediction = augmentation(image_prediction)
            image_predictions.append(image_prediction)

        return torch.mean(torch.stack(image_predictions), dim=0)
        


        