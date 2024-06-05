import os
from dotenv import load_dotenv
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import requests
import json


load_dotenv()

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

PROMPTS = []


def get_response(prompt, past_responses=None):
    url = "http://localhost:11434/api/chat"
    if past_responses is None:
        history = []
    else:
        history = [
            {"role": "assistant", "content": message} for message in past_responses
        ]
    history.append({"role": "user", "content": prompt})

    data = {
        "model": "llama3:70b",
        "messages": history,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        print(response.json())
        print(f"Request failed with status code {response.status_code}")
        return None

def generate_image(prompt):
    torch.cuda.empty_cache()
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

    torch.cuda.empty_cache()
    image = pipe(prompt=prompt, num_inference_steps=25).images[0]
    image.save(f"./outputs/{prompt[:50].replace(' ', '_')}.png")

def main():
    while True:
        past_prompts = PROMPTS[-10:] if len(PROMPTS) >= 10 else PROMPTS
        new_prompt = get_response(f"Please provide a new image prompt.  Please do not use any of the following existing prompts: {', '.join(past_prompts)}")

        if new_prompt:
            PROMPTS.append(new_prompt)
            generate_image(new_prompt)
        else:
            print("Failed to get a new prompt. Trying again...")


if __name__ == "__main__":
    prompt = "Galactic cityscape"
    torch.cuda.empty_cache()
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

    torch.cuda.empty_cache()
    image = pipe(prompt=prompt, num_inference_steps=25).images[0]
    image.save(f"./outputs/{prompt[:50].replace(' ', '_')}.png")
    # main()