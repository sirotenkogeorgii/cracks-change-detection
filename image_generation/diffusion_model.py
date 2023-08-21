import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from huggingface_hub import hf_hub_download, HfApi
from helpers import get_random_string
from collections import namedtuple
from accelerate import Accelerator
from configurations import CONFIG
from psutil import virtual_memory
import torch
import os

ModelFiles = namedtuple("ModelFiles", 
            ["model_path", 
            "model_config", 
            "model_index", 
            "scheduler_config", 
            "repo_id"], 
            defaults=[None])


class UNetDiffusion:
    def __init__(
        self, 
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: tuple[int] = (128, 128, 256, 256, 512, 512),
        down_block_types: tuple[str] = None,
        up_block_types: tuple[str] = None,
        model_files: ModelFiles = None,
        scheduler = None
    ) -> None:

        available_ram = virtual_memory().total / 1e9
        if available_ram < 20:
            raise RuntimeError("You do not have enough available RAM: {:.1f} gigabytes. You need at least 20 gigabytes.".format(available_ram))

        self.device = ("cuda" if torch.cuda.is_available() else "mps"
            if torch.backends.mps.is_available() else "cpu")

        self.noise_scheduler = scheduler
        if scheduler is None:
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=CONFIG.num_train_timesteps)

        if down_block_types is None:
                down_block_types = (
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                )

        if up_block_types is None:
                up_block_types = (
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D"
                )

        model = UNet2DModel(
            sample_size=sample_size,  # the target image resolution
            in_channels=in_channels,  # the number of input channels, 3 for RGB images
            out_channels=out_channels,  # the number of output channels
            layers_per_block=layers_per_block,#2  # how many ResNet layers to use per UNet block
            block_out_channels=block_out_channels,  # the number of output channes for each UNet block
            down_block_types=down_block_types,
            up_block_types=up_block_types
        )

        if model_files is not None:
            if model_files.repo_id is not None:
                hf_hub_download(repo_id=model_files.repo_id, filename=model_files.model_path, local_dir=CONFIG.pretrained_dir)
                hf_hub_download(repo_id=model_files.repo_id, filename=model_files.model_config, local_dir=CONFIG.pretrained_dir)
                hf_hub_download(repo_id=model_files.repo_id, filename=model_files.model_index, local_dir=CONFIG.pretrained_dir)
                hf_hub_download(repo_id=model_files.repo_id, filename=model_files.scheduler_config, local_dir=CONFIG.pretrained_dir)
            local_model_path = "/".join(model_files.model_path.split('/')[:-1])
            model = model.from_pretrained(f"{CONFIG.pretrained_dir}/{local_model_path}", token=CONFIG.read_token)
        
        self._model = model.to(self.device)


    def _save_images(self, images, repo_id: str, commit_message: str) -> None:
        api = HfApi()

        samples_dir = os.path.join(f"{CONFIG.output_dir}/samples")
        os.makedirs(samples_dir, exist_ok=True)

        for image in images:
            file_id = get_random_string(10)
            local_filename = f"{samples_dir}/generated_image_{file_id}.png"
            image.save(local_filename)
            if repo_id is not None:
                api.upload_file(
                        path_or_fileobj=local_filename,
                        path_in_repo=f"/generated_data/generated_image_{file_id}.png",
                        repo_id=repo_id,
                        token=CONFIG.write_token,
                        commit_message=commit_message)


    def generate_images(
        self, 
        samples_num: int, 
        repo_id: str = None,
        commit_message: str = None,
        is_pretrained_pipeline: bool = False, 
        seed: int = None
        ) -> None:

        accelerator = Accelerator(
        mixed_precision=CONFIG.mixed_precision,
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(CONFIG.output_dir, "logs")
        )

        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(self._model), scheduler=self.noise_scheduler)
        if is_pretrained_pipeline:
            pipeline = pipeline.from_pretrained(CONFIG.pretrained_dir, from_tf=True)

        if seed is None: seed = CONFIG.seed
        generated_images = pipeline(batch_size=samples_num, generator=torch.manual_seed(seed)).images
        self._save_images(generated_images, repo_id, commit_message)   

    def get_model(self): return self._model

    def parameters(self): return self._model.parameters() 

    def __call__(self, noisy_images, timesteps, return_dict=False): return self._model(noisy_images, timesteps, return_dict=return_dict)
