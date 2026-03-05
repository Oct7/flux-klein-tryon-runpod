# FLUX.2-Klein Virtual Try-On API

FLUX.2-klein-9B (fp8) + Virtual Try-On LoRA 기반의 I2I(Image-to-Image) FastAPI 서버입니다.
사람 이미지 + 상의 + 하의 3장을 입력하면 가상 피팅 합성 이미지를 반환합니다

## 모델 정보

- **베이스 모델**: `black-forest-labs/FLUX.2-klein-9b-fp8` (9B 파라미터, fp8 양자화)
- **LoRA**: `fal/flux-klein-9b-virtual-tryon-lora` (`flux-klein-tryon.safetensors`, ~127MB)
- **파이프라인**: `Flux2KleinPipeline` (diffusers)

## 빠른 시작

### 원클릭 배포 (권장)

```bash
# .env 파일 설정 (선택)
echo "API_KEY=your-secret-key" > .env
echo "TARGET_GPU_IDS=0" >> .env

# 배포 스크립트 실행
chmod +x setup.sh && ./setup.sh
```

### 수동 빌드 & 실행

```bash
# 이미지 빌드
sudo docker build -t flux-klein-tryon-api .

# 컨테이너 실행
sudo docker run -d \
  --name flux-klein-tryon-api-container \
  --restart=always \
  --gpus all \
  -e API_KEY="your-secret-key" \
  -e TARGET_GPU_IDS="0" \
  -p 8000:8000 \
  flux-klein-tryon-api
```

### 다중 GPU 실행 예시

```bash
# GPU 0, 1 동시 사용
sudo docker run -d \
  --gpus all \
  -e TARGET_GPU_IDS="0,1" \
  -e API_KEY="your-secret-key" \
  -p 8000:8000 \
  flux-klein-tryon-api
```

## API 사용 가이드

### 인증

모든 엔드포인트에 `X-API-Key` 헤더가 필요합니다.

```
X-API-Key: your-secret-key
```

---

### `POST /generate` — Virtual Try-On 이미지 생성

#### 요청 스키마

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `prompt` | string | ✅ | — | 프롬프트 (TRYON 트리거 워드 포함 권장) |
| `person_image` | ImageInput | ✅ | — | 사람 이미지 |
| `top_image` | ImageInput | ✅ | — | 상의 이미지 |
| `bottom_image` | ImageInput | ✅ | — | 하의 이미지 |
| `height` | int | ❌ | 1024 | 출력 이미지 높이 |
| `width` | int | ❌ | 1024 | 출력 이미지 너비 |
| `num_inference_steps` | int | ❌ | 28 | 추론 스텝 수 |
| `guidance_scale` | float | ❌ | 2.5 | CFG 스케일 |
| `lora_scale` | float | ❌ | 1.0 | LoRA 가중치 |
| `seed` | int | ❌ | -1 | 시드 (-1=랜덤) |
| `upload_url` | string | ❌ | null | S3 presigned URL (지정 시 URL 반환) |

`ImageInput` 구조:
```json
{
  "base64": "data:image/png;base64,...",  // base64 이미지
  "url": "https://..."                     // 또는 이미지 URL
}
```
`base64` 또는 `url` 중 하나 필수.

#### 응답

**S3 업로드 시:**
```json
{
  "status_code": 200,
  "image_url": "https://s3.amazonaws.com/bucket/image.png",
  "seed": 12345678
}
```

**base64 폴백:**
```json
{
  "status_code": 200,
  "image_base64": "data:image/png;base64,...",
  "seed": 12345678
}
```

#### curl 예시 (URL 방식)

```bash
curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "TRYON a person wearing the outfit",
    "person_image": {"url": "https://example.com/person.jpg"},
    "top_image":    {"url": "https://example.com/top.jpg"},
    "bottom_image": {"url": "https://example.com/bottom.jpg"},
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 2.5,
    "seed": -1
  }'
```

#### curl 예시 (base64 방식)

```bash
PERSON_B64=$(base64 -w 0 person.jpg)
TOP_B64=$(base64 -w 0 top.jpg)
BOTTOM_B64=$(base64 -w 0 bottom.jpg)

curl -X POST http://localhost:8000/generate \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"TRYON a person wearing the outfit\",
    \"person_image\": {\"base64\": \"$PERSON_B64\"},
    \"top_image\":    {\"base64\": \"$TOP_B64\"},
    \"bottom_image\": {\"base64\": \"$BOTTOM_B64\"}
  }"
```

---

### `GET /status` — 서버 상태 확인

```bash
curl http://localhost:8000/status -H "X-API-Key: your-secret-key"
```

```json
{
  "status_code": 200,
  "status": "ready",
  "active_requests": 0,
  "total_worker_count": 1,
  "idle_worker_count": 1
}
```

---

### `GET /status/gpu` — GPU 하드웨어 모니터링

```bash
curl http://localhost:8000/status/gpu -H "X-API-Key: your-secret-key"
```

```json
{
  "status_code": 200,
  "total_gpu_count": 1,
  "active_worker_count": 1,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "is_active_serving": true,
      "memory": {"total_mib": 81920.0, "used_mib": 24576.0, "utilization_percent": 30.0},
      "utilization": {"gpu_percent": 85, "memory_percent": 30},
      "temperature_c": 42,
      "power_usage_w": 280.5
    }
  ]
}
```

## 프롬프트 가이드

Virtual Try-On LoRA는 `TRYON` 트리거 워드를 인식합니다.

```
TRYON a person wearing the outfit
TRYON model wearing casual clothing
TRYON 옷을 입은 사람
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `API_KEY` | `your-secret-key-1234` | API 인증 키 |
| `TARGET_GPU_IDS` | `0` | 사용할 GPU ID (쉼표 구분, 예: `0,1`) |

## 모델 사전 다운로드 (선택)

빌드 없이 로컬에서 사전 다운로드하려면:

```bash
pip install diffusers huggingface_hub
python download_model.py
```

## 로그 확인

```bash
sudo docker logs -f flux-klein-tryon-api-container
```

---

## fal.ai 서버리스 배포

Docker/GPU 서버 관리 없이 fal.ai 서버리스 GPU 함수로 배포합니다.
`fal_app.py` 하나로 인프라 없는 자동 스케일링 API를 제공합니다.

### fal.ai vs Docker 비교

| 항목 | Docker (app.py) | fal.ai (fal_app.py) |
|------|-----------------|----------------------|
| 인프라 관리 | 직접 서버 관리 필요 | 불필요 (serverless) |
| 스케일링 | 수동 | 자동 (scale-to-zero) |
| 비용 | 24/7 서버 비용 | 요청 시에만 과금 |
| 인증 | `X-API-Key` 헤더 | fal 내장 인증 |
| 이미지 반환 | base64 / S3 presigned URL | fal CDN URL (자동) |
| GPU 모니터링 | `/status/gpu` 엔드포인트 | fal dashboard |
| 동시 처리 | `ThreadPoolExecutor` | `max_concurrency` 설정 |

### 배포 방법

```bash
# 1. fal CLI 설치
pip install fal

# 2. fal 로그인
fal auth login

# 3. 로컬 테스트
fal run fal_app.py::FluxKleinTryOn

# 4. 프로덕션 배포
fal deploy fal_app.py::FluxKleinTryOn --app-name "flux-klein-tryon"
```

### fal.ai API 스키마

#### 요청 (TryOnInput)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `prompt` | string | `"TRYON a person wearing the outfit"` | 프롬프트 (TRYON 트리거 워드 포함 권장) |
| `person_image_url` | string | 필수 | 사람 이미지 URL |
| `garment_image_urls` | list[string] | 필수 | 옷 이미지 URL 리스트 (1장 이상) |
| `height` | int | 1024 | 출력 이미지 높이 (256~2048) |
| `width` | int | 1024 | 출력 이미지 너비 (256~2048) |
| `num_inference_steps` | int | 28 | 추론 스텝 수 (1~50) |
| `guidance_scale` | float | 2.5 | CFG 스케일 (0.0~20.0) |
| `lora_scale` | float | 1.0 | LoRA 가중치 (0.0~2.0) |
| `seed` | int \| null | null | 시드 (null = 랜덤) |

#### 응답 (TryOnOutput)

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | Image | 생성된 이미지 (fal CDN URL) |
| `seed` | int | 사용된 시드 |

#### curl 예시

```bash
curl -X POST https://fal.run/YOUR_USER_ID/flux-klein-tryon \
  -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "TRYON a person wearing the outfit",
    "person_image_url": "https://example.com/person.jpg",
    "garment_image_urls": [
      "https://example.com/top.jpg",
      "https://example.com/bottom.jpg"
    ],
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 2.5,
    "seed": null
  }'
```

#### 응답 예시

```json
{
  "image": {
    "url": "https://fal.media/files/...",
    "content_type": "image/png",
    "width": 1024,
    "height": 1024
  },
  "seed": 123456789
}
```

### fal.ai 앱 설정

`fal_app.py`의 `FluxKleinTryOn` 클래스 상단에서 설정을 변경할 수 있습니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `machine_type` | `"GPU-H100"` | GPU 타입 (H100 80GB) |
| `keep_alive` | `300` | 워커 유지 시간 (초) |
| `min_concurrency` | `0` | 최소 워커 수 (0 = scale-to-zero) |
| `max_concurrency` | `2` | 최대 동시 워커 수 |

---

## RunPod 서버리스 배포

fal.ai 대신 **RunPod Serverless**로 배포합니다.
Docker 이미지를 빌드해 RunPod 엔드포인트로 등록하면 자동 스케일링 GPU API를 운영할 수 있습니다.

### 3종 배포 방식 비교

| 항목 | Docker (`app.py`) | fal.ai (`fal_app.py`) | RunPod (`runpod_app.py`) |
|------|-------------------|----------------------|--------------------------|
| 인프라 관리 | 직접 서버 관리 | 불필요 (serverless) | 불필요 (serverless) |
| 스케일링 | 수동 | 자동 (scale-to-zero) | 자동 (scale-to-zero) |
| 비용 | 24/7 서버 비용 | 요청 시에만 과금 | 요청 시에만 과금 |
| 인증 | `X-API-Key` 헤더 | fal 내장 인증 | RunPod API 키 |
| 이미지 반환 | base64 / S3 URL | fal CDN URL (자동) | base64 / S3 presigned URL |
| Cold Start 최소화 | N/A | fal `keep_alive` | 모델 이미지 bake |
| 입력 형식 | JSON (base64/URL) | URL 전용 | JSON (base64/URL) |

### 빌드 & 푸시

```bash
# 1. Docker 이미지 빌드
docker build -f Dockerfile.runpod -t flux-klein-tryon-runpod .

# 2. Docker Hub 또는 RunPod Container Registry에 푸시
docker tag flux-klein-tryon-runpod your-dockerhub-user/flux-klein-tryon-runpod:latest
docker push your-dockerhub-user/flux-klein-tryon-runpod:latest
```

### RunPod 엔드포인트 생성

1. [RunPod Console](https://www.runpod.io/console/serverless) 접속
2. **Serverless** → **+ New Endpoint** 클릭
3. 설정:
   - **Container Image**: `your-dockerhub-user/flux-klein-tryon-runpod:latest`
   - **GPU**: A100 80GB 또는 H100 80GB (모델 크기 고려)
   - **Min Workers**: `0` (scale-to-zero)
   - **Max Workers**: 원하는 최대 동시 처리 수
   - **Idle Timeout**: `300` (초)
4. **Deploy** 클릭 → 엔드포인트 ID 확인 (예: `abc123xyz`)

### API 사용

RunPod Serverless API는 비동기 방식입니다.

#### 작업 제출

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "TRYON a person wearing the outfit",
      "person_image": "https://example.com/person.jpg",
      "garment_images": [
        "https://example.com/top.jpg",
        "https://example.com/bottom.jpg"
      ],
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 28,
      "guidance_scale": 2.5,
      "seed": -1
    }
  }'
```

응답 예시:
```json
{"id": "job-abc123", "status": "IN_QUEUE"}
```

#### 결과 조회

```bash
curl https://api.runpod.ai/v2/{ENDPOINT_ID}/status/job-abc123 \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

응답 예시:
```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "image_base64": "data:image/png;base64,...",
    "seed": 123456789
  }
}
```

#### 동기 방식 (runsync — 60초 이내 응답 시)

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "TRYON a person wearing the outfit",
      "person_image": "https://example.com/person.jpg",
      "garment_images": ["https://example.com/top.jpg"],
      "seed": 42
    }
  }'
```

#### S3 presigned URL 업로드 (선택)

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "TRYON a person wearing the outfit",
      "person_image": "https://example.com/person.jpg",
      "garment_images": ["https://example.com/top.jpg"],
      "upload_url": "https://s3.amazonaws.com/your-bucket/result.png?X-Amz-Signature=..."
    }
  }'
```

응답:
```json
{"image_url": "https://s3.amazonaws.com/your-bucket/result.png", "seed": 987654321}
```

### RunPod Input 스키마

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `prompt` | string | `"TRYON a person wearing the outfit"` | 프롬프트 (TRYON 트리거 워드 포함 권장) |
| `person_image` | string | **필수** | 사람 이미지 (base64 또는 URL) |
| `garment_images` | list[string] | **필수** | 옷 이미지 리스트 (base64/URL, 1장 이상) |
| `height` | int | `1024` | 출력 이미지 높이 |
| `width` | int | `1024` | 출력 이미지 너비 |
| `num_inference_steps` | int | `28` | 추론 스텝 수 |
| `guidance_scale` | float | `2.5` | CFG 스케일 |
| `lora_scale` | float | `1.0` | LoRA 가중치 |
| `seed` | int | `-1` | 시드 (-1 = 랜덤) |
| `upload_url` | string | `null` | S3 presigned URL (지정 시 URL 반환) |

### 로컬 테스트 (`--rp_serve_api`)

RunPod SDK의 로컬 API 서버 기능으로 배포 전 검증할 수 있습니다.

```bash
# 의존성 설치
pip install runpod

# 로컬 API 서버 실행 (localhost:8000)
python runpod_app.py --rp_serve_api

# 별도 터미널에서 curl 테스트
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "TRYON a person wearing the outfit",
      "person_image": "https://example.com/person.jpg",
      "garment_images": ["https://example.com/top.jpg"]
    }
  }'
```
