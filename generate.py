import sys
import os
import torch
from pathlib import Path
sys.path.append("sdxl/sdxl_modules")
sys.path.append("sdxl/SDXL_Test")
from sdxl_modules import downloader, pipe_loader, face_analysis, style_manager

PIPE = None
APP = None
STYLES = None

def setup_environment():
    global PIPE, APP, STYLES

    if not Path("SDXL_Test/models/YamerMIX_v8").exists():
        downloader.download_models()

    if PIPE is None:
        print("[INFO] Pipeline yükleniyor...")
        controlnet_path = "/content/SDXL_Test/models/instantid/ControlNetModel"
        PIPE = pipe_loader.load_pipe(
            base_model="/content/SDXL_Test/models/YamerMIX_v8",
            controlnet=pipe_loader.load_controlnet(controlnet_path)
        )
        pipe_loader.load_ip_adapter(PIPE, "/content/SDXL_Test/models/instantid/ip-adapter.bin")
    else:
        print("[CACHE] Pipeline zaten yüklü, tekrar yüklenmedi.")

    if APP is None:
        APP = face_analysis.setup_face_app()

    if STYLES is None:
        STYLES = style_manager.load_styles()

    return PIPE , APP, STYLES


def generate_images(image_path, prompt_text, style_name, num_images=1, outdir="SDXL_Test/images"):
    global PIPE, APP, STYLES

    if PIPE is None or APP is None or STYLES is None:
        PIPE, APP, STYLES = setup_environment()

    face_image, face_emb, face_kps = face_analysis.get_face_info(APP, image_path)

    style = next(s for s in STYLES if s["name"] == style_name)
    styled_prompt, negative_prompt = style_manager.get_style_prompt(style, prompt_text)

    os.makedirs(outdir, exist_ok=True)
    results = []

    output_images = PIPE(
        prompt=styled_prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.6,
        ip_adapter_scale=0.7,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=768,
        width=768,
        num_images_per_prompt=num_images
    ).images

    for idx, img in enumerate(output_images):
        save_path = os.path.join(outdir, f"output_{idx+1}.png")
        img.save(save_path)
        results.append(save_path)
        print(f"[INFO] Kaydedildi: {save_path}")

    return results
