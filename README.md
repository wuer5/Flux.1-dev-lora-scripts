# Flux.1-dev-lora-scripts

It is easy for you to LoRA your Flux.1-dev using this script!

This repository provides a Flux.1-dev LoRA training script modified from the ```xflux``` and ```diffusers``` repositories.

You only need the ```diffusers``` library for Flux.1-dev LoRA training. We will release other fine-tuning methods, such as ControlNet.

## Configuration

Settings are configured in `train.yaml`.

Note that running Flux.1-dev LoRA training requires approximately 40GB of CUDA memory with a batch size of 1. You can set `gradient_checkpointing=True` to reduce CUDA memory usage.

## RUN
```
accelerate launch train_flux_lora.py
```
