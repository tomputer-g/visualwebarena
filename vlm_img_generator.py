"""Memory-grid image generators for GUI agent trajectories.

Two implementations share the ``ImgGenerator`` ABC:

- ``LocalImgGenerator`` — pure-PIL renderer, no API calls.
- ``VLMImgGenerator`` — Gemini image generation via Vertex AI.
  Requires the ``google-genai`` package (``pip install google-genai``).

  Environment variables (Vertex AI):
  - ``PROJECT_ID`` — GCP project ID (required).
  - ``LOCATION`` — GCP region (optional, default ``us-central1``).

Both maintain a 2×4 grid of ``patch_px``-sized cells. ``generate_img`` updates
cell ``step_num % 8`` and returns the full grid image.
"""

from __future__ import annotations

import base64
import io
import os
import textwrap
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

from google import genai  # type: ignore[import-untyped]
from google.genai import types  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont


_GRID_ROWS = 2
_GRID_COLS = 4
_NUM_CELLS = _GRID_ROWS * _GRID_COLS


@dataclass
class MemoryStep:
    intent: str
    dom_tree: str
    cot: str
    action: str


class ImgGenerator(ABC):
    """Base class: maintains a 2×4 memory grid; subclasses implement ``_generate_patch``."""

    def __init__(self, patch_px: int = 256) -> None:
        self.patch_px = patch_px
        self.memory_img = Image.new(
            "RGB",
            (_GRID_COLS * patch_px, _GRID_ROWS * patch_px),
            color=(255, 255, 255),  # type: ignore[arg-type]
        )

    def generate_img(
        self,
        step_num: int,
        intent: str,
        dom_tree: str,
        cot: str,
        action: str,
    ) -> Image.Image:
        step = MemoryStep(intent=intent, dom_tree=dom_tree, cot=cot, action=action)
        patch = self._generate_patch(step)
        row, col = divmod(step_num % _NUM_CELLS, _GRID_COLS)
        self.memory_img.paste(patch, (col * self.patch_px, row * self.patch_px))
        return self.memory_img

    @abstractmethod
    def _generate_patch(self, step: MemoryStep) -> Image.Image: ...


class LocalImgGenerator(ImgGenerator):
    """Renders each memory patch locally with PIL (no API calls)."""

    def __init__(
        self,
        patch_px: int = 256,
        font_size: int = 7,
        padding: int = 8,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (30, 30, 30),
        label_color: Tuple[int, int, int] = (80, 80, 200),
    ) -> None:
        self.font_size = font_size
        self.padding = padding
        self.bg_color = bg_color
        self.text_color = text_color
        self.label_color = label_color
        super().__init__(patch_px)

    def _generate_patch(self, step: MemoryStep) -> Image.Image:
        img = Image.new("RGB", (self.patch_px, self.patch_px), color=self.bg_color)  # type: ignore[arg-type]
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", self.font_size
            )
            bold = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", self.font_size
            )
        except OSError:
            font = bold = ImageFont.load_default()

        usable = self.patch_px - 2 * self.padding
        chars_per_line = max(1, usable // (self.font_size * 6 // 10))
        y = self.padding

        def draw_section(label: str, body: str) -> None:
            nonlocal y
            if y + self.font_size > self.patch_px - self.padding:
                return
            draw.text((self.padding, y), label, font=bold, fill=self.label_color)
            y += self.font_size + 4
            for raw_line in body.splitlines():
                for line in textwrap.wrap(raw_line, width=chars_per_line) or [""]:
                    if y + self.font_size > self.patch_px - self.padding:
                        return
                    draw.text((self.padding, y), line, font=font, fill=self.text_color)
                    y += self.font_size + 2
            y += self.font_size

        draw_section("Intent:", step.intent)
        draw_section("CoT:", step.cot)
        draw_section("Action:", step.action)
        draw_section("SOM:", step.dom_tree)
        return img


class VLMImgGenerator(ImgGenerator):
    """Generates each memory patch via Gemini image generation on Vertex AI."""

    def __init__(
        self,
        model_id: str = "publishers/google/models/gemini-2.5-flash-image",
        aspect_ratio: str = "1:1",
        patch_px: int = 256,
        max_dom_chars: int = 8000,
    ) -> None:
        self.model_id = model_id
        self.aspect_ratio = aspect_ratio
        self.max_dom_chars = max_dom_chars
        self._client: Optional[genai.Client] = None
        super().__init__(patch_px)

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            project = os.environ.get("PROJECT_ID")
            if not project:
                raise RuntimeError(
                    "PROJECT_ID environment variable is required for Vertex AI image generation."
                )
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=os.environ.get("LOCATION", "us-central1"),
            )
        return self._client

    def _compose_prompt(self, step: MemoryStep) -> str:
        dom = (
            step.dom_tree
            if len(step.dom_tree) <= self.max_dom_chars
            else step.dom_tree[: self.max_dom_chars] + "\n\n[... DOM truncated ...]"
        )
        body = (
            f"## Task intent\n{step.intent}\n\n"
            f"## Page / DOM (accessibility tree excerpt)\n{dom}\n\n"
            f"## Agent chain-of-thought\n{step.cot}\n\n"
            f"## Chosen action\n{step.action}"
        )
        return (
            "Generate an image that displays the following text exactly as written, "
            "as if it were printed on a white background with clear, readable typography:\n\n"
            f"{body}\n"
        )

    def _generate_patch(self, step: MemoryStep) -> Image.Image:
        response = self._ensure_client().models.generate_content(
            model=self.model_id.rstrip("/").split("/")[-1],
            contents=self._compose_prompt(step),
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=self.aspect_ratio),
            ),
        )

        if not response.candidates:
            raise RuntimeError("Gemini image response has no candidates.")

        image_bytes: Optional[bytes] = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                data = part.inline_data.data
                image_bytes = base64.b64decode(data) if isinstance(data, str) else data
                break

        if image_bytes is None:
            raise RuntimeError("No image part in Gemini response (expected inline_data image).")

        patch = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return patch.resize((self.patch_px, self.patch_px), Image.Resampling.LANCZOS)  # type: ignore[attr-defined]
