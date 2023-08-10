from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128*4 #2500 // 4  # the generated image resolution
    train_batch_size = 2 # 1
    eval_batch_size = 3*3  # how many images to sample during evaluation
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    num_train_timesteps = 1000
    save_image_epochs = 50
    save_model_epochs = 50
    write_token = "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"
    read_token = "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"
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "concrete_generation"  # the model namy locally and on the HF Hub
    pretrained_dir = "/content/pretrained_model"
    pretrained = True

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


CONFIG = TrainingConfig()