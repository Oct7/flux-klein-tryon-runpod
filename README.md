# FLUX.2-Klein Virtual Try-On — RunPod Serverless API

FLUX.2-klein-9B + Virtual Try-On LoRA 기반 가상 피팅 API입니다.
사람 이미지 + 옷 이미지(1장 이상)를 입력하면 피팅 결과 이미지를 반환합니다.

## 모델 정보

- **베이스 모델**: `black-forest-labs/FLUX.2-klein-9B` (9B 파라미터, bfloat16)
- **LoRA**: `fal/flux-klein-9b-virtual-tryon-lora` (`flux-klein-tryon.safetensors`)
- **파이프라인**: `Flux2KleinPipeline` (diffusers)
- **인프라**: RunPod Serverless (자동 스케일링, scale-to-zero)

---

## 빠른 시작

### 작업 제출

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "TRYON a young female model. Replace ONLY the upper body only with navy blue fleece half-zip pullover as shown in the reference image. Keep the pants and shoes unchanged. The final image is a full body shot.",
      "person_image": "https://example.com/person.jpg",
      "garment_images": ["https://example.com/top.jpg"],
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 4,
      "guidance_scale": 2.5,
      "seed": -1
    }
  }'
```

응답:
```json
{"id": "job-abc123", "status": "IN_QUEUE"}
```

### 결과 조회

```bash
curl https://api.runpod.ai/v2/{ENDPOINT_ID}/status/job-abc123 \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "image_base64": "data:image/png;base64,...",
    "seed": 42
  }
}
```

---

## Input 스키마

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `person_image` | string | **필수** | 사람 이미지 (base64 또는 URL) |
| `garment_images` | list[string] | **필수** | 옷 이미지 리스트 (base64/URL, 1장 이상) |
| `prompt` | string | `"TRYON a person wearing the outfit"` | 프롬프트 (아래 가이드 참고) |
| `height` | int | `1024` | 출력 이미지 높이 |
| `width` | int | `1024` | 출력 이미지 너비 |
| `num_inference_steps` | int | `28` | 추론 스텝 수 (4=빠름, 28=고품질) |
| `guidance_scale` | float | `2.5` | CFG 스케일 |
| `lora_scale` | float | `1.0` | LoRA 가중치 |
| `seed` | int | `-1` | 시드 (-1 = 랜덤) |
| `upload_url` | string | `null` | S3 presigned URL (지정 시 URL 반환) |

## Output 스키마

**base64 반환 (기본):**
```json
{
  "image_base64": "data:image/png;base64,...",
  "seed": 123456789
}
```

**S3 업로드 시 (`upload_url` 지정):**
```json
{
  "image_url": "https://s3.amazonaws.com/bucket/result.png",
  "seed": 123456789
}
```

---

## 프롬프트 가이드

### 권장 패턴 (v3)

입력한 옷만 정확하게 변경하려면 변경 부위를 명시하고 나머지 유지를 지시합니다.

```
TRYON [person description].
Replace ONLY the [part] with [garment description] as shown in the reference image.
Keep [other parts] unchanged.
The final image is a full body shot.
```

### 부위별 프롬프트 예시

**상의 변경 (top / jacket)**
```
TRYON a young female model with braided hair.
Replace ONLY the upper body only with navy blue fleece half-zip pullover as shown in the reference image.
Keep the pants and shoes unchanged.
The final image is a full body shot.
```

**아우터 변경 (coat / outer)**
```
TRYON a young male model with short dark hair.
Replace ONLY the outerwear only with camel brown fluffy oversized teddy fur coat as shown in the reference image.
Keep the inner top, pants and shoes unchanged.
The final image is a full body shot.
```

**하의 변경 (pants / skirt)**
```
TRYON a slim female model with glasses.
Replace ONLY the lower body only with dark navy wide-leg cargo denim pants as shown in the reference image.
Keep the top and shoes unchanged.
The final image is a full body shot.
```

### TRYON 트리거 워드

LoRA는 `TRYON` 트리거 워드를 인식합니다. 프롬프트 앞에 반드시 포함하세요.

---

## 이미지 입력 권장사항

### 리사이즈 전처리 (권장)

입력 이미지의 **긴 변을 1024px로 리사이즈** 후 전송하면 성능이 향상됩니다.

| | 원본 | 리사이즈 1024 |
|---|---|---|
| 페이로드 크기 | ~4.3MB | ~370KB (-91%) |
| 제출 시간 | 2.7s | 0.7s |
| 추론 시간 | ~5s | ~3.4s |

```python
from PIL import Image
import io, base64

def img_to_b64_resized(path, max_size=1024):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
```

---

## 성능 벤치마크

워커 3개 기준, 8개 병렬 요청:

| steps | 개별 추론 | 8개 전체 |
|-------|----------|---------|
| 28    | ~11s     | ~60s    |
| 4     | ~3.5s    | ~20s    |

> steps=4 권장 (품질 대비 속도 최적)

---

## Python 예제

```python
import base64, io, time, requests
from PIL import Image

ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-runpod-api-key"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def img_to_b64(path, max_size=1024):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# 작업 제출
payload = {
    "input": {
        "prompt": (
            "TRYON a young female model. "
            "Replace ONLY the upper body only with navy blue fleece half-zip pullover "
            "as shown in the reference image. "
            "Keep the pants and shoes unchanged. "
            "The final image is a full body shot."
        ),
        "person_image": img_to_b64("person.jpg"),
        "garment_images": [img_to_b64("top.png")],
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 2.5,
        "seed": 42,
    }
}

res = requests.post(f"{BASE_URL}/run", json=payload, headers=headers)
job_id = res.json()["id"]
print(f"Job: {job_id}")

# 결과 폴링
while True:
    time.sleep(5)
    r = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers).json()
    status = r.get("status")
    print(f"Status: {status}")
    if status == "COMPLETED":
        output = r["output"]
        b64 = output["image_base64"].split(",", 1)[1]
        with open("result.png", "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"저장 완료: result.png (seed={output['seed']})")
        break
    elif status in ("FAILED", "CANCELLED"):
        print(f"실패: {r.get('error')}")
        break
```

---

## 배포 구조

```
GitHub Push (main)
    ↓
GitHub Actions (build-runpod.yml)
    ↓
ghcr.io/oct7/flux-klein-tryon-runpod:latest
    ↓
RunPod Serverless Endpoint
```

### 환경 변수 (RunPod 콘솔 설정)

| 변수 | 설명 |
|------|------|
| `HF_TOKEN` | HuggingFace API 토큰 (게이티드 모델 접근용) |

---

## 헬스 체크

```bash
curl https://api.runpod.ai/v2/{ENDPOINT_ID}/health \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

```json
{
  "workers": {"idle": 3, "ready": 3, "running": 0},
  "jobs": {"completed": 0, "inQueue": 0}
}
```
