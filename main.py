import os
from dotenv import load_dotenv
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

load_dotenv()

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

pipe.load_lora_weights('nworb-ucsb/face_LoRA',)
_ = pipe.to("cuda")

prompt = "happy" # @param

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save("output_image.png")