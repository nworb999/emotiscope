import os
from dotenv import load_dotenv
import openai
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import requests
from PIL import Image
import json
import time

torch.cuda.empty_cache()
print(torch.cuda.is_available())

load_dotenv()

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')
client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))


PROMPTS = []

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights('nworb-ucsb/face_LoRA',)
pipe = pipe.to("cuda")


def get_response_ollama(prompt, past_responses=None):
    url = "http://localhost:11434/api/chat"
    if past_responses is None:
        history = []
    else:
        history = [
            {"role": "assistant", "content": message} for message in past_responses
        ]
    history.append({"role": "user", "content": prompt})

    data = {
        "model": "mixtral:latest",
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
    
def get_response(prompt, past_responses=None):
    time.sleep(4)
    if past_responses is None:
        past_responses = []

    messages = [{"role": "user", "content": message} for message in past_responses]
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Ensure you specify the correct model, e.g., gpt-3.5-turbo if needed
        messages=messages,
    )

    return response.choices[0].message.content


def generate_image(prompt):
    torch.cuda.empty_cache()
    image = pipe(prompt=prompt[:50], num_inference_steps=25).images[0]
    low_res_image = image.resize((256, 256), Image.LANCZOS)

    low_res_image.save(f"./outputs/{prompt[:50].replace(' ', '_').replace('_"', '').replace('"')}.png")

def main():
    while True:
        past_prompts = PROMPTS[-5:] if len(PROMPTS) >= 10 else PROMPTS
        new_prompt = get_response(f"Please provide a new image prompt involving a face or facial expression.  Please do not use any of the following existing prompts: {', '.join(past_prompts)}")

        if new_prompt:
            PROMPTS.append(new_prompt)
            generate_image(new_prompt)
        else:
            print("Failed to get a new prompt. Trying again...")


if __name__ == "__main__":
    main()
