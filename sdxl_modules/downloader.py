from huggingface_hub import login, snapshot_download
import os
import subprocess

def login_hf(token: str):
    login(token)

def download_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    models = {
        "YamerMIX_v8": "wangqixun/YamerMIX_v8",
        "instantid": "InstantX/InstantID"
    }

    for name, repo in models.items():
        snapshot_download(
            repo_id=repo,
            local_dir=f"models/{name}",
            local_dir_use_symlinks=False
        )

    if not os.path.exists("InstantID"):
        subprocess.run([
            "git", "clone", "https://github.com/InstantID/InstantID.git", "InstantID"
        ])
    subprocess.run([
        "cp",
        "InstantID/pipeline_stable_diffusion_xl_instantid.py",
        "./"
    ])

    if not os.path.exists("models/instantid/ip-adapter.bin"):
        subprocess.run([
            "wget",
            "-O", "models/instantid/ip-adapter.bin",
            "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"
        ])