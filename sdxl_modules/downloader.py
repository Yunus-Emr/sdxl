from huggingface_hub import login, snapshot_download
import os
import subprocess

def login_hf(token: str):
    login(token)

def download_models():
    os.makedirs("sdxl/models", exist_ok=True)
    os.makedirs("sdxl/images", exist_ok=True)

    models = {
        "YamerMIX_v8": "wangqixun/YamerMIX_v8",
        "instantid": "InstantX/InstantID"
    }

    for name, repo in models.items():
        snapshot_download(
            repo_id=repo,
            local_dir=f"sdxl/models/{name}",
            local_dir_use_symlinks=False
        )

    if not os.path.exists("sdxl/InstantID"):
        subprocess.run([
            "git", "clone", "https://github.com/InstantID/InstantID.git", "sdxl/InstantID"
        ])
    subprocess.run([
        "cp",
        "sdxl/InstantID/pipeline_stable_diffusion_xl_instantid.py",
        "sdxl/"
    ])

    if not os.path.exists("sdxl/models/instantid/ip-adapter.bin"):
        subprocess.run([
            "wget",
            "-O", "sdxl/models/instantid/ip-adapter.bin",
            "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"
        ])