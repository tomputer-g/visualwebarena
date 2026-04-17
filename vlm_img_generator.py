"""Memory-grid image generator for GUI agent trajectories (Vertex / Gemini image).

Requires the ``google-genai`` package (``pip install google-genai``).

Environment variables (Vertex AI, same convention as ``img_gen/generate_image.py``):

- ``PROJECT_ID`` — GCP project ID (required).
- ``LOCATION`` — GCP region (optional, default ``us-central1``).

Each call to :meth:`VLMImgGenerator.generate_img` fills one cell of a fixed 2×4 grid
with a new patch from Gemini image generation, representing intent, DOM state,
chain-of-thought, and action at that step.

``step_num`` is a **zero-based** step index: cell ``step_num % 8`` is updated
(steps 0–7 fill the grid left-to-right, top row then bottom row; step 8 wraps to cell 0).
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

from google import genai
from google.genai import types
from PIL import Image


_GRID_ROWS = 2
_GRID_COLS = 4
_NUM_CELLS = _GRID_ROWS * _GRID_COLS


class VLMImgGenerator:
    """Maintains a 2×4 memory image; each cell holds one Gemini-generated memory patch."""

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

        self.memory_img_h: int
        self.memory_img_w: int
        self.memory_patch_h: int
        self.memory_patch_w: int

        self._init_memory_img(patch_px)

    def _init_memory_img(self, patch_px: int) -> None:
        """
        Initialize the memory image: a 2×4 grid of patch-sized cells.
        Each patch i is an image representing the agent's state at step i (mod 8).
        """
        self.memory_patch_w = patch_px
        self.memory_patch_h = patch_px
        self.memory_img_w = _GRID_COLS * self.memory_patch_w
        self.memory_img_h = _GRID_ROWS * self.memory_patch_h
        self.memory_img = Image.new(
            "RGB", (self.memory_img_w, self.memory_img_h), color=(255, 255, 255)
        )

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            project = os.environ.get("PROJECT_ID")
            if not project:
                raise RuntimeError(
                    "PROJECT_ID environment variable is required for Vertex AI image generation."
                )
            location = os.environ.get("LOCATION", "us-central1")
            self._client = genai.Client(
                vertexai=True, project=project, location=location
            )
        return self._client

    def _model_short_name(self) -> str:
        return self.model_id.rstrip("/").split("/")[-1]

    def _compose_memory_prompt(
        self, intent: str, dom_tree: str, cot: str, action: str
    ) -> str:
        dom = dom_tree if len(dom_tree) <= self.max_dom_chars else (
            dom_tree[: self.max_dom_chars] + "\n\n[... DOM truncated ...]"
        )
        text = (
            f"## Task intent\n{intent}\n\n"
            f"## Page / DOM (accessibility tree excerpt)\n{dom}\n\n"
            f"## Agent chain-of-thought\n{cot}\n\n"
            f"## Chosen action\n{action}"
        )
        return (
            "Generate an image that displays the following text exactly as written, "
            "as if it were printed on a white background with clear, readable typography:\n\n"
            f"{text}\n"
        )

    def generate_img(
        self,
        step_num: int,
        intent: str,
        dom_tree: str,
        cot: str,
        action: str,
    ) -> Image.Image:
        """
        Generate a memory patch for this step, paste it into cell ``step_num % 8``,
        and return the full memory image.

        ``step_num`` is zero-based (first GUI step uses ``0``).
        """
        prompt = self._compose_memory_prompt(intent, dom_tree, cot, action)
        patch = self._generate_memory_patch(prompt)
        cell_index = step_num % _NUM_CELLS
        self._update_memory_img(cell_index, patch)
        return self.memory_img

    def _generate_memory_patch(self, prompt: str) -> Image.Image:
        """Call Gemini (Vertex) image generation with the given prompt; return a resized RGB patch."""
        client = self._ensure_client()
        model_name = self._model_short_name()

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
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
                if isinstance(data, str):
                    data = base64.b64decode(data)
                image_bytes = data
                break

        if image_bytes is None:
            raise RuntimeError(
                "No image part in Gemini response (expected inline_data image)."
            )

        patch = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        patch = patch.resize(
            (self.memory_patch_w, self.memory_patch_h), Image.Resampling.LANCZOS
        )
        return patch

    def _update_memory_img(self, i: int, patch: Image.Image) -> None:
        """Paste ``patch`` into cell index ``i`` (row-major 2×4 grid)."""
        if self.memory_img is None:
            raise RuntimeError("memory_img is not initialized.")
        row, col = divmod(i, _GRID_COLS)
        x0 = col * self.memory_patch_w
        y0 = row * self.memory_patch_h
        self.memory_img.paste(patch, (x0, y0))
