import io
import os
import gc
import time
import base64
import traceback
import torch
import requests
from PIL import Image
from diffusers import Flux2KleinPipeline
import runpod

# 글로벌 파이프라인 (워커 시작 시 1회 로드)
pipe = None

# RunPod Network Volume 마운트 경로 (없으면 HuggingFace 기본 캐시 사용)
# 콘솔에서 Volume을 /runpod-volume 로 마운트하면 자동으로 사용됨
_VOLUME_PATH = "/runpod-volume/models"
if os.path.isdir("/runpod-volume"):
    os.environ.setdefault("HF_HOME", _VOLUME_PATH)
    os.makedirs(_VOLUME_PATH, exist_ok=True)

MAX_LOAD_RETRIES = 3
LOAD_RETRY_DELAY = 10


def load_model():
    """모델 + LoRA 로드 (워커 시작 시 1회 실행, 실패 시 최대 3회 재시도)

    - Network Volume 마운트 시: /runpod-volume/models 캐시 사용 (빠름)
    - Volume 없을 시: HF_HOME 또는 기본 캐시 사용
    """
    global pipe
    print(f"HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

    for attempt in range(1, MAX_LOAD_RETRIES + 1):
        try:
            print(f"Loading FLUX.2-Klein pipeline... (attempt {attempt}/{MAX_LOAD_RETRIES})")
            pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9b-fp8",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to("cuda")

            print("Loading Virtual Try-On LoRA weights...")
            pipe.load_lora_weights(
                "fal/flux-klein-9b-virtual-tryon-lora",
                weight_name="flux-klein-tryon.safetensors",
            )
            print("Model + LoRA loaded successfully!")
            return
        except Exception as e:
            print(f"Model load attempt {attempt} failed: {e}")
            if attempt < MAX_LOAD_RETRIES:
                pipe = None
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(LOAD_RETRY_DELAY)
            else:
                raise


def cleanup_gpu():
    """매 job 후 GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_image(src: str) -> Image.Image:
    """base64 또는 URL에서 PIL Image를 로드합니다."""
    if src.startswith("data:"):
        # data:image/...;base64,<data>
        b64_data = src.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
    elif not src.startswith("http://") and not src.startswith("https://"):
        # 순수 base64 문자열
        img_bytes = base64.b64decode(src)
    else:
        # URL 다운로드
        response = requests.get(src, timeout=30)
        response.raise_for_status()
        img_bytes = response.content
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def handler(job):
    """RunPod 서버리스 핸들러

    Input 스키마 (job["input"]):
        prompt          : str        — 프롬프트 (TRYON 트리거 워드 포함 권장)
        person_image    : str (필수) — base64 또는 URL
        garment_images  : list[str]  — base64/URL 리스트 (1장 이상, 필수)
        height          : int        — 출력 높이 (기본 1024)
        width           : int        — 출력 너비 (기본 1024)
        num_inference_steps : int    — 스텝 수 (기본 28)
        guidance_scale  : float      — CFG (기본 2.5)
        lora_scale      : float      — LoRA 가중치 (기본 1.0)
        seed            : int        — -1=랜덤 (기본 -1)
        upload_url      : str|null   — S3 presigned URL (선택)

    Output:
        {"image_base64": "data:image/png;base64,...", "seed": int}
        또는 upload_url 지정 시:
        {"image_url": "https://...", "seed": int}
    """
    job_input = job["input"]

    # ---- 입력 검증 — raise로 RunPod FAILED 처리 ----
    person_image_src = job_input.get("person_image")
    if not person_image_src:
        raise ValueError("person_image is required (base64 or URL)")

    garment_images_src = job_input.get("garment_images")
    if (
        not garment_images_src
        or not isinstance(garment_images_src, list)
        or len(garment_images_src) == 0
    ):
        raise ValueError("garment_images is required and must be a non-empty list")

    # ---- 파라미터 파싱 ----
    prompt = job_input.get("prompt", "TRYON a person wearing the outfit")
    height = int(job_input.get("height", 1024))
    width = int(job_input.get("width", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 28))
    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    lora_scale = float(job_input.get("lora_scale", 1.0))
    seed = int(job_input.get("seed", -1))
    upload_url = job_input.get("upload_url", None)

    try:
        # ---- 이미지 로드 ----
        person_img = load_image(person_image_src)
        garment_imgs = [load_image(src) for src in garment_images_src]
        reference_images = [person_img] + garment_imgs

        # ---- 시드 처리 ----
        if seed == -1:
            seed = torch.seed() % (2**32)
        generator = torch.Generator("cuda").manual_seed(seed)

        # ---- 추론 ----
        output = pipe(
            prompt=prompt,
            image=reference_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            joint_attention_kwargs={"scale": lora_scale},
            generator=generator,
        )
        result_image = output.images[0]

        # ---- 결과 직렬화 ----
        img_io = io.BytesIO()
        result_image.save(img_io, format="PNG")
        img_bytes = img_io.getvalue()

        # S3 presigned URL 업로드 (선택)
        if upload_url:
            upload_res = requests.put(
                upload_url,
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=60,
            )
            upload_res.raise_for_status()
            pure_url = upload_url.split("?")[0]
            return {"image_url": pure_url, "seed": seed}

        # base64 반환 (기본)
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "image_base64": f"data:image/png;base64,{encoded}",
            "seed": seed,
        }

    except Exception as e:
        print(f"Handler error: {traceback.format_exc()}")
        raise  # RunPod가 FAILED로 처리

    finally:
        cleanup_gpu()


# 워커 시작 시 모델 로드 후 서버리스 루프 진입
load_model()
runpod.serverless.start({"handler": handler})
