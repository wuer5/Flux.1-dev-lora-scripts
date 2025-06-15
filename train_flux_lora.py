#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Callable
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FluxControlPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from dataset import MyDataset


# Copy from `https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py`
def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, _, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def set_flux_transformer_lora(flux_transformer, args):
    target_modules = [
        "x_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    )
    flux_transformer = PeftModel(
        flux_transformer,
        transformer_lora_config,
        adapter_name="flux_transformer_lora_adapter",
    )
    flux_transformer.print_trainable_parameters()
    return flux_transformer


logger = get_logger(__name__)


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (
        pixel_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="A flux.1-dev training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="train.yml",
        help="path to config",
    )
    args = parser.parse_args()

    return args.config


def main(args):

    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError(
            "`gaussian` LoRA init scheme isn't supported when `use_lora_bias` is True."
        )

    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=str(logging_out_dir)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models loaded successfully")

    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    vae.to(dtype=torch.float32)  # keep the VAE in float32.
    flux_transformer.to(dtype=weight_dtype, device=accelerator.device)

    # set lora to flux_transformer
    flux_transformer = set_flux_transformer_lora(flux_transformer, args)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [flux_transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimization parameters
    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, flux_transformer.parameters())
    )
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare dataset and dataloader.
    train_dataset = MyDataset(
        dataset_json_path=args.dataset_json_path, resluotion=args.resolution
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * num_update_steps_per_epoch
            * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != args.max_train_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
    )

    # If use fixed_prompts, we will pre-process prompt_embeds
    if args.fixed_prompts:
        text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = (
                text_encoding_pipeline.encode_prompt(args.fixed_prompts, prompt_2=None)
            )
        del text_encoding_pipeline
        free_memory()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    timesteps = get_schedule(
        999,
        (1024 // 8) * (1024 // 8) // 4,
        shift=True,
    )
    timesteps = torch.tensor(timesteps)
    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # Convert images to latent space
                # vae encode
                img = batch["img"].to(accelerator.device)
                x1 = encode_images(img, vae.to(accelerator.device), weight_dtype)

                if args.offload:
                    # offload vae to CPU.
                    vae.cpu()

                bsz, c, h, w = x1.shape
                x0 = torch.randn_like(x1, device=accelerator.device, dtype=weight_dtype)
                indices = (torch.rand((bsz,)) * len(timesteps)).long()
                t = timesteps[indices].to(device=accelerator.device, dtype=weight_dtype)

                # Note this is different from the paper, using the way of DDPM
                xt = (1 - t) * x1 + x0

                # pack the latents.
                xt = _pack_latents(
                    xt,
                    batch_size=bsz,
                    num_channels_latents=c,
                    height=h,
                    width=w,
                )

                # latent image ids for RoPE.
                latent_image_ids = _prepare_latent_image_ids(
                    bsz,
                    h // 2,
                    w // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if accelerator.unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=xt.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                if not args.fixed_prompts:
                    # text encoding.
                    captions = batch["captions"]
                    text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                    with torch.no_grad():
                        prompt_embeds, pooled_prompt_embeds, text_ids = (
                            text_encoding_pipeline.encode_prompt(
                                captions, prompt_2=None
                            )
                        )
                    if args.offload:
                        text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                # Predict.
                model_pred = flux_transformer(
                    hidden_states=xt,
                    timestep=t,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = _unpack_latents(
                    model_pred,
                    height=h * vae_scale_factor,
                    width=w * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # Note this is different from the paper, using the way of DDPM
                # flow-matching loss
                target = x0 - x1
                loss = torch.mean(
                    ((model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    or accelerator.is_main_process
                ):
                    if global_step % args.checkpointing_steps == 0:
                        # Save checkpoint
                        checkpoint_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        lora_weight_path = os.path.join(
                            args.output_dir, f"lora_weight-{global_step}"
                        )
                        accelerator.save_state(checkpoint_path)
                        logger.info(f"Saved state to {checkpoint_path}")

                        accelerator.unwrap_model(flux_transformer).save_pretrained(
                            lora_weight_path
                        )
                        logger.info(f"Saved lora weight to {lora_weight_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_weight_path = os.path.join(args.output_dir, f"lora_weight-{global_step}")
        accelerator.unwrap_model(flux_transformer).save_pretrained(lora_weight_path)
        logger.info(f"Saved lora weight to {lora_weight_path}")

        del flux_transformer
        del text_encoding_pipeline
        del vae
        free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    args = OmegaConf.load(parse_args())
    main(args)
