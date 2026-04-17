from __future__ import annotations

from pathlib import Path
from typing import Any


class HfQwenVlGenerator:
    """Lazy Hugging Face Qwen2.5-VL runner with optional LoRA adapter."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        *,
        bf16: bool = True,
        max_pixels: int | None = 512 * 512,
        device_map: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.bf16 = bf16
        self.max_pixels = max_pixels
        self.device_map = device_map
        self.processor: Any | None = None
        self.model: Any | None = None
        self._torch: Any | None = None
        self._process_vision_info: Any | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing Qwen2.5-VL inference dependencies. Install: "
                "transformers peft accelerate qwen-vl-utils torch"
            ) from exc

        self._torch = torch
        self._process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Missing PEFT dependency. Install: peft") from exc
            adapter = Path(self.adapter_path)
            if not adapter.exists():
                raise FileNotFoundError(f"LoRA adapter path not found: {adapter}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter))

        self.model.eval()

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        *,
        max_new_tokens: int = 256,
    ) -> str:
        self.load()
        assert self.processor is not None
        assert self.model is not None
        assert self._torch is not None
        assert self._process_vision_info is not None

        content: list[dict[str, Any]] = []
        if image_path:
            image_item: dict[str, Any] = {
                "type": "image",
                "image": str(image_path),
            }
            if self.max_pixels:
                image_item["max_pixels"] = self.max_pixels
            content.append(image_item)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if image_path:
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated[:, prompt_len:]
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
