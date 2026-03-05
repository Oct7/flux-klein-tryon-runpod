import os
import sys
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import hf_hub_download

hf_home = os.environ.get("HF_HOME", "/app/model_cache")
os.environ["HF_HOME"] = hf_home
os.makedirs(hf_home, exist_ok=True)

try:
    print("Downloading and caching FLUX.2-klein-9B base model...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    print("Base model downloaded successfully!")
    del pipe  # 메모리 해제

    print("Downloading Virtual Try-On LoRA weights...")
    hf_hub_download(
        repo_id="fal/flux-klein-9b-virtual-tryon-lora",
        filename="flux-klein-tryon.safetensors",
    )
    print("LoRA weights downloaded successfully!")
    print("All model artifacts cached. Ready for serving.")
except Exception as e:
    print(f"FATAL: model download failed: {e}")
    sys.exit(1)
