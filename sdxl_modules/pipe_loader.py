import sys
sys.path.append("sdxl/sdxl_modules")
sys.path.append("sdxl")
import torch
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline


def load_controlnet(controlnet_path, dtype=torch.float16):
    return ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)

def load_pipe(base_model, controlnet, dtype=torch.float16, device="cuda"):
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    return pipe

def load_ip_adapter(pipe, face_adapter_path):
    pipe.load_ip_adapter_instantid(face_adapter_path)