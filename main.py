from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
import uvicorn 
import json

from modules.models import Txt2ImgRequest 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

device = "cuda"
model_id_sd15 = "CompVis/stable-diffusion-v1-5"

base = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
)
base.to("cuda")
refiner = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

@app.post("/v1/api/sdxl1/txt2img")
async def txt2img(body: Txt2ImgRequest):
    with autocast(device): 
        image = base(
            prompt=body.prompt,
            num_inference_steps=body.steps,
            denoising_end=0.8,
            output_type="latent",
        ).images
        image = refiner(
            prompt=body.prompt,
            num_inference_steps=body.steps,
            denoising_start=0.8,
            image=image,
        ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data = { "image": imgstr }
    json_str = json.dumps(data, indent=4, default=str)
    return Response(content=json_str, media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)