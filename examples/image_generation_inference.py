import sys
import pathlib
sys.path.append(f"{str(pathlib.Path(__file__).parent.parent.resolve())}/image_generation")

from configurations import CONFIG
from diffusion_model import ModelFiles, UNetDiffusion
from huggingface_hub import login

login(token=CONFIG.read_token, add_to_git_credential=True)

# CONFIG.pretrained_dir = 
model_files = ModelFiles(
    model_path="299/unet/diffusion_pytorch_model.bin",
    model_config="299/unet/config.json",
    model_index="model_index.json",
    scheduler_config="scheduler/scheduler_config.json",
    repo_id="georgiisirotenko/concrete_generation"
)

model = UNetDiffusion(CONFIG.image_size, 3, 3, 4, model_files=model_files)
model.generate_images(10, "georgiisirotenko/concrete_generation", "some commit message", False)
