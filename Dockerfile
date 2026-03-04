# 베이스 이미지: 공식 PyTorch 이미지 (A100/CUDA 12.4 최적화)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 1. 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA 환경 변수
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /app

# 2. Python 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. 플래시 어텐션(선택 사항) - GPU 환경에서 빌드 시 주석 해제
# RUN pip install flash-attn --no-build-isolation

# 4. 애플리케이션 소스 복사
COPY download_model.py .
COPY app.py .

# 5. 모델 사전 다운로드 (용량 문제로 빌드 시 주석 처리 — 서버 시작 시 자동 다운로드)
# RUN python download_model.py

# 6. 포트 노출
EXPOSE 8000

# 7. 서버 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
