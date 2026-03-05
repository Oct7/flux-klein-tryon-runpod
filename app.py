from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import io
import torch
import base64
import os
import uuid
import asyncio
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from diffusers import Flux2KleinPipeline
import pynvml
import queue

# 보안 설정: 환경 변수에서 API_KEY를 가져오며, 기본값은 임시로 지정 (배포 시 변경 권장)
API_KEY = os.getenv("API_KEY", "your-secret-key-1234")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(header_value: str = Depends(api_key_header)):
    if header_value == API_KEY:
        return header_value
    raise HTTPException(
        status_code=403, detail="Could not validate credentials - Invalid API Key"
    )

app = FastAPI(title="FLUX.2-Klein Virtual Try-On API")

# 전역 상태 및 동시 처리 설정
pipes = {}  # {gpu_id: pipeline}
# TARGET_GPU_IDS 예: "0,1" -> [0, 1]
gpu_id_str = os.getenv("TARGET_GPU_IDS", os.getenv("TARGET_GPU_ID", "0"))
TARGET_GPU_IDS = [int(x.strip()) for x in gpu_id_str.split(",")]

# 동시 처리를 위한 워커 수 설정 (GPU 개수만큼 병렬 처리)
num_workers = len(TARGET_GPU_IDS)
executor = ThreadPoolExecutor(max_workers=num_workers)
active_requests = 0

# 사용 가능한 GPU 인덱스를 담는 큐
available_gpus = queue.Queue()
for gid in TARGET_GPU_IDS:
    available_gpus.put(gid)

@app.on_event("startup")
def startup_event():
    # pynvml 초기화
    try:
        pynvml.nvmlInit()
        print(f"NVML initialized. Monitoring {pynvml.nvmlDeviceGetCount()} GPUs.")
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
    load_models_to_gpus()

def load_models_to_gpus():
    global pipes
    print(f"Starting to load models on GPUs: {TARGET_GPU_IDS}")

    for gid in TARGET_GPU_IDS:
        print(f"[{gid}] Checking CUDA availability...")
        if not torch.cuda.is_available():
            print(f"ERROR: CUDA not available for GPU {gid}")
            continue

        print(f"[{gid}] Loading FLUX.2-Klein pipeline...")
        try:
            pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
            print(f"[{gid}] Moving model to cuda:{gid}...")
            pipe.to(f"cuda:{gid}")

            print(f"[{gid}] Loading Virtual Try-On LoRA weights...")
            pipe.load_lora_weights(
                "fal/flux-klein-9b-virtual-tryon-lora",
                weight_name="flux-klein-tryon.safetensors",
            )
            print(f"[{gid}] LoRA loaded successfully!")

            pipes[gid] = pipe
            print(f"[{gid}] Model + LoRA loaded successfully on GPU {gid}!")
        except Exception as e:
            print(f"[{gid}] Failed to load on GPU {gid}: {e}")

    if not pipes:
        raise RuntimeError("No GPUs were initialized successfully.")
    print(f"Total {len(pipes)} GPU workers are ready!")


# ---- 요청 스키마 ----

class ImageInput(BaseModel):
    base64: Optional[str] = None   # base64 인코딩된 이미지 (data:image/... 접두어 포함 가능)
    url: Optional[str] = None      # 이미지 다운로드 URL

class GenerateRequest(BaseModel):
    prompt: str                            # TRYON 트리거 워드 포함 프롬프트
    person_image: ImageInput               # 사람 이미지 (필수)
    garment_images: List[ImageInput]       # 옷 이미지 리스트 (1장 이상, 필수)
    height: int = 1024                     # 출력 높이
    width: int = 1024                      # 출력 너비
    num_inference_steps: int = 28          # LoRA 권장 스텝 수
    guidance_scale: float = 2.5            # LoRA 권장 CFG
    lora_scale: float = 1.0               # LoRA 가중치 스케일
    seed: int = -1                         # -1 = 랜덤
    upload_url: Optional[str] = None       # S3 presigned URL (선택)


# ---- 이미지 로딩 헬퍼 ----

def load_image_from_input(image_input: ImageInput) -> Image.Image:
    """base64 또는 URL에서 PIL Image를 로드합니다."""
    if image_input.base64:
        b64_data = image_input.base64
        # data:image/...;base64, 접두어 제거
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif image_input.url:
        response = requests.get(image_input.url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        raise ValueError("ImageInput에는 base64 또는 url 중 하나가 필수입니다.")


# ---- 상태 엔드포인트 ----

@app.get("/status")
def get_status(api_key: str = Depends(get_api_key)):
    global available_gpus, pipes, active_requests
    if not pipes:
        return {"status_code": 200, "status": "loading", "active_requests": active_requests}

    idle_count = available_gpus.qsize()
    status = "ready" if idle_count > 0 else "busy"

    return {
        "status_code": 200,
        "status": status,
        "active_requests": active_requests,
        "total_worker_count": len(pipes),
        "idle_worker_count": idle_count,
    }

@app.get("/status/gpu")
def get_gpu_status(api_key: str = Depends(get_api_key)):
    """서버에 장착된 모든 GPU의 상세 하드웨어 상태를 리스트로 반환합니다."""
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_list = []
        active_serving_count = 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            is_active_serving = (i in TARGET_GPU_IDS)
            if is_active_serving:
                active_serving_count += 1

            gpu_list.append({
                "index": i,
                "name": name,
                "is_active_serving": is_active_serving,
                "memory": {
                    "total_mib": round(mem_info.total / (1024**2), 2),
                    "used_mib": round(mem_info.used / (1024**2), 2),
                    "utilization_percent": round(mem_info.used / mem_info.total * 100, 1),
                },
                "utilization": {
                    "gpu_percent": utilization.gpu,
                    "memory_percent": utilization.memory,
                },
                "temperature_c": temp,
                "power_usage_w": round(power_usage, 2),
            })

        return {
            "status_code": 200,
            "total_gpu_count": device_count,
            "active_worker_count": active_serving_count,
            "gpus": gpu_list,
        }
    except Exception as e:
        return {
            "status_code": 500,
            "error": str(e),
            "message": "Failed to fetch multi-GPU stats",
        }


# ---- 생성 태스크 ----

def _generate_task(req: GenerateRequest):
    """실제로 GPU에서 Virtual Try-On 추론을 수행하는 동기 함수 (Worker Thread에서 실행됨)"""
    global pipes, available_gpus

    # 사용 가능한 GPU 하나 꺼내기 (비어있을 경우 스레드가 여기서 대기함)
    gpu_id = available_gpus.get()
    device = f"cuda:{gpu_id}"
    pipe = pipes[gpu_id]

    try:
        if req.seed == -1:
            seed = torch.seed() % (2**32)
            generator = torch.Generator(device).manual_seed(seed)
        else:
            seed = req.seed
            generator = torch.Generator(device).manual_seed(seed)

        # 이미지 로드
        person_img = load_image_from_input(req.person_image)
        garment_imgs = [load_image_from_input(g) for g in req.garment_images]

        # 사람 + 옷 이미지(들)을 순서대로 전달
        reference_images = [person_img] + garment_imgs

        output = pipe(
            prompt=req.prompt,
            image=reference_images,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            joint_attention_kwargs={"scale": req.lora_scale},
            generator=generator,
        )
        image = output.images[0]

        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_bytes = img_io.getvalue()

        if req.upload_url:
            upload_res = requests.put(
                req.upload_url, data=img_bytes, headers={"Content-Type": "image/png"}
            )
            upload_res.raise_for_status()
            pure_url: str = req.upload_url.split("?")[0]
            return {"status_code": 200, "image_url": pure_url, "seed": seed}

        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "status_code": 200,
            "image_base64": f"data:image/png;base64,{encoded_img}",
            "seed": seed,
        }

    except Exception as e:
        return {"status_code": 500, "error": str(e), "message": "Internal processing error"}
    finally:
        # 어떤 상황에서도 GPU 반납
        available_gpus.put(gpu_id)


@app.post("/generate")
async def generate_image(req: GenerateRequest, api_key: str = Depends(get_api_key)):
    global pipes, active_requests
    if not pipes:
        raise HTTPException(status_code=503, detail="Models are still loading")

    active_requests += 1
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, _generate_task, req)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status_code": 500,
                "error": str(e),
                "message": "Failed to generate image or upload to server",
            },
        )
    finally:
        active_requests -= 1
