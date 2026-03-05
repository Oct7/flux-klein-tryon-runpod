import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image, download_file
from typing import Optional


# ---- Input/Output 스키마 ----

class TryOnInput(BaseModel):
    prompt: str = Field(
        default="TRYON a person wearing the outfit",
        description="프롬프트 (TRYON 트리거 워드 포함 권장)",
    )
    person_image_url: str = Field(description="사람 이미지 URL")
    garment_image_urls: list[str] = Field(description="옷 이미지 URL 리스트 (1장 이상)")
    height: int = Field(default=1024, ge=256, le=2048)
    width: int = Field(default=1024, ge=256, le=2048)
    num_inference_steps: int = Field(default=28, ge=1, le=50)
    guidance_scale: float = Field(default=2.5, ge=0.0, le=20.0)
    lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    seed: Optional[int] = Field(default=None, description="None = 랜덤")


class TryOnOutput(BaseModel):
    image: Image = Field(description="생성된 이미지 (fal CDN URL)")
    seed: int = Field(description="사용된 시드")


# ---- fal.ai 서버리스 앱 ----

class FluxKleinTryOn(fal.App):
    machine_type = "GPU-H100"
    keep_alive = 300
    min_concurrency = 0
    max_concurrency = 2
    app_name = "flux-klein-tryon"

    requirements = [
        "torch==2.5.1",
        "torchvision",
        "git+https://github.com/huggingface/diffusers.git",
        "git+https://github.com/huggingface/transformers.git",
        "accelerate>=0.34.0",
        "safetensors",
        "sentencepiece",
        "tiktoken",
        "peft",
        "Pillow",
        "huggingface_hub",
        "hf_transfer",
    ]

    def setup(self):
        """워커 시작시 1회 — 모델 + LoRA 로드"""
        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        ).to("cuda")

        self.pipe.load_lora_weights(
            "fal/flux-klein-9b-virtual-tryon-lora",
            weight_name="flux-klein-tryon.safetensors",
        )

    @fal.endpoint("/")
    def generate(self, input: TryOnInput) -> TryOnOutput:
        """Virtual Try-On 이미지 생성"""
        import torch
        from PIL import Image as PILImage

        # 시드 처리
        if input.seed is None:
            seed = torch.seed() % (2**32)
        else:
            seed = input.seed
        generator = torch.Generator("cuda").manual_seed(seed)

        # 이미지 다운로드
        person_img = PILImage.open(download_file(input.person_image_url)).convert("RGB")
        garment_imgs = [
            PILImage.open(download_file(url)).convert("RGB")
            for url in input.garment_image_urls
        ]
        reference_images = [person_img] + garment_imgs

        # 추론
        output = self.pipe(
            prompt=input.prompt,
            image=reference_images,
            height=input.height,
            width=input.width,
            num_inference_steps=input.num_inference_steps,
            guidance_scale=input.guidance_scale,
            joint_attention_kwargs={"scale": input.lora_scale},
            generator=generator,
        )

        # 결과 반환 (fal CDN 자동 업로드)
        result_image = Image.from_pil(output.images[0], format="png")
        return TryOnOutput(image=result_image, seed=seed)
