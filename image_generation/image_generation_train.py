from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfFolder, Repository, whoami, login
from diffusion_model import UNetDiffusion
from torchvision import transforms
from diffusers import DDPMPipeline
from accelerate import Accelerator
from configurations import CONFIG
from datasets import load_dataset
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import torch
import os


def _make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


@torch.no_grad() # I added it.
def _evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    image_grid = _make_grid(images, rows=3, cols=3)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def _get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def _train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = _get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, model.noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = model.noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=model.noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                _evaluate(config, epoch, pipeline)
                pipeline.save_pretrained(config.output_dir)
                if config.push_to_hub:
                  repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)


def main() -> None:
    model = UNetDiffusion(CONFIG.image_size, 3, 3, 4)
    model.generate_images(10, "georgiisirotenko/concrete_generation", "some commit message", False)

    CONFIG.dataset_name = "georgiisirotenko/inner_patches"
    dataset = load_dataset(CONFIG.dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}
    dataset.set_transform(transform)

    first_partition = len(dataset) // 2
    second_partition = len(dataset) - first_partition
    dataset, _ = torch.utils.data.random_split(dataset, lengths=[first_partition, second_partition])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG.train_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CONFIG.lr_warmup_steps,
        num_training_steps=(len(train_dataloader)*CONFIG.num_epochs),
    )

    _train_loop(CONFIG, model, optimizer, train_dataloader, lr_scheduler)


if __name__ == "__main__":
    login(token=CONFIG.write_token, add_to_git_credential=True)
    main()


    for i in range(12):
        print(123)




        