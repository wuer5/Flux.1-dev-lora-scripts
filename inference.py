from diffusers import FluxPipeline
from peft import PeftModel
import torch

pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
adapter_path = "[YOUR LORA PATH]"
pipe = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.bfloat16,
)

pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer, adapter_path, adapter_name="flux_transformer_lora_adapter"
)

prompt = ["YOUR PROMPT"]
image = pipe(prompt, num_inference_steps=40).images[0]
image.save("./output.png")
