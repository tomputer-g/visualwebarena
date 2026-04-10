from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union
import numpy as np
import faiss

if TYPE_CHECKING:
    from PIL import Image as PILImage

# Flexible image type: PIL Image, raw bytes, file path string, or XML/HTML string
ImageLike = Union["PILImage.Image", bytes, str, np.ndarray]


@dataclass
class Image:
    """Wraps an image in whatever form the embedding model expects.

    For BLIP-2/Flan-T5 pipelines the canonical form is a PIL Image, but raw
    bytes, file paths, and XML/HTML strings are also accepted so callers are
    not forced to convert upfront.
    """
    data: ImageLike


@dataclass
class Action:
    """A single agent action represented as a string."""
    value: str


@dataclass
class Embedding:
    """A dense vector produced by an embedding model (e.g. BLIP-2/Flan-T5).

    Stored as a 1-D float32 NumPy array so it can be fed directly into
    similarity search routines (cosine, dot-product, FAISS, etc.).
    """
    vector: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.ndim != 1:
            raise ValueError(
                f"Embedding must be 1-D, got shape {self.vector.shape}"
            )

    @property
    def dim(self) -> int:
        return self.vector.shape[0]


@dataclass
class Trajectory:
    """A single (screenshot, task, action) triple used as a memory entry."""
    screenshot: Image
    task_description: str
    next_action: Action

    @staticmethod
    def from_raw(_raw: object) -> Trajectory:
        # TODO: parse raw input into Trajectory fields
        raise NotImplementedError


class EmbeddingModel:
    """Wraps BLIP-2/Flan-T5 to embed a Trajectory into a dense vector.

    Architecture path:
      screenshot  ──► ViT encoder ──► Q-Former ──► language projection ─┐
      task text   ──► T5 token embeddings ──────────────────────────────┤
                                                                         ▼
                                                           T5 encoder hidden states
                                                           (mean-pooled → Embedding)

    The resulting vector captures both the visual state and the task context,
    making it suitable for cosine-similarity retrieval of oracle actions.
    """

    DEFAULT_MODEL = "Salesforce/blip2-flan-t5-xl"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        import torch  # type: ignore[import-untyped]
        from transformers import Blip2ForConditionalGeneration, Blip2Processor  # type: ignore[import-untyped]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor: Blip2Processor = Blip2Processor.from_pretrained(model_name)
        self._model: Blip2ForConditionalGeneration = (
            Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
            .to(device)
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, trajectory: Trajectory) -> Embedding:
        """Return a mean-pooled T5 encoder embedding for a Trajectory.

        The image features (via ViT + Q-Former) are prepended to the task
        text token embeddings before running the T5 encoder, so the resulting
        vector is jointly conditioned on both modalities.
        """
        import torch  # type: ignore[import-untyped]

        pil_image = self._load_image(trajectory.screenshot)
        text = trajectory.task_description

        inputs = self._processor(
            images=pil_image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # 1. Vision encoder (ViT)
            vision_out = self._model.vision_model(
                pixel_values=inputs.pixel_values,
                return_dict=True,
            )
            image_embeds = vision_out.last_hidden_state  # (1, n_patches, vision_dim)
            image_attn = torch.ones(
                image_embeds.shape[:-1], dtype=torch.long, device=self.device
            )

            # 2. Q-Former: cross-attends learned query tokens over image features
            query_tokens = self._model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            qformer_out = self._model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn,
                return_dict=True,
            )
            query_hidden = qformer_out.last_hidden_state  # (1, n_queries, qformer_dim)

            # 3. Project Q-Former output into T5 embedding space
            lang_inputs = self._model.language_projection(query_hidden)  # (1, n_queries, t5_dim)
            lang_attn = torch.ones(
                lang_inputs.shape[:-1], dtype=torch.long, device=self.device
            )

            # 4. T5 token embeddings for the task text
            text_embeds = self._model.language_model.get_input_embeddings()(
                inputs.input_ids
            )  # (1, text_len, t5_dim)

            # 5. Concatenate vision tokens + text tokens → T5 encoder
            combined_embeds = torch.cat([lang_inputs, text_embeds], dim=1)
            combined_attn = torch.cat([lang_attn, inputs.attention_mask], dim=1)

            encoder_out = self._model.language_model.encoder(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attn,
                return_dict=True,
            )
            hidden = encoder_out.last_hidden_state  # (1, seq_len, t5_dim)

            # 6. Masked mean-pool over the full attended sequence
            mask = combined_attn.unsqueeze(-1).float()  # (1, seq_len, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, t5_dim)

        vec: np.ndarray = pooled.squeeze(0).cpu().float().numpy()
        return Embedding(vector=vec)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Image) -> "PILImage.Image":
        """Convert any ImageLike variant into a PIL RGB Image."""
        from PIL import Image as PILImage

        data = image.data
        if isinstance(data, PILImage.Image):
            return data.convert("RGB")
        if isinstance(data, np.ndarray):
            return PILImage.fromarray(data).convert("RGB")
        if isinstance(data, bytes):
            return PILImage.open(io.BytesIO(data)).convert("RGB")
        if isinstance(data, str):
            # Could be a file path or an HTML/XML string — try path first
            candidate = Path(data)
            if candidate.exists():
                return PILImage.open(candidate).convert("RGB")
            raise ValueError(
                f"EmbeddingModel._load_image: string data is not a valid file path: {data!r}"
            )
        raise TypeError(f"Unsupported image type: {type(data)}")


class OracleActionMemory:
    """FAISS-backed store mapping embeddings to oracle actions.

    Similarity is cosine (inner product on L2-normalised vectors).
    Persistence uses two sidecar files:
      <path>.faiss  — the FAISS index
      <path>.json   — the parallel list of action strings

    An optional EmbeddingModel can be attached so that add() / query() accept
    Trajectory objects directly instead of pre-computed Embedding objects.
    """

    _INDEX_SUFFIX = ".faiss"
    _ACTIONS_SUFFIX = ".json"

    def __init__(self, dim: int, embedding_model: EmbeddingModel | None = None) -> None:
        self._dim = dim
        # IndexFlatIP + pre-normalisation == cosine similarity
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._actions: list[str] = []
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, source: Embedding | Trajectory, action: Action | None = None) -> None:
        """Insert one entry into the database.

        Accepted call signatures:
          memory.add(embedding, action)   — pre-computed embedding
          memory.add(trajectory)          — auto-embed via EmbeddingModel;
                                            action is taken from trajectory.next_action
        """
        embedding, action = self._resolve(source, action)
        assert action is not None  # guaranteed by require_action=True (default)
        vec = self._normalise(embedding.vector)
        self._index.add(vec[np.newaxis, :])
        self._actions.append(action.value)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, source: Embedding | Trajectory, k: int = 3) -> list[Action]:
        """Return the top-k most similar actions for the given embedding or trajectory."""
        if len(self._actions) == 0:
            return []
        embedding, _ = self._resolve(source, action=None, require_action=False)
        k = min(k, len(self._actions))
        vec = self._normalise(embedding.vector)
        _, indices = self._index.search(vec[np.newaxis, :], k)
        return [Action(self._actions[i]) for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Write the index and action list to <path>.faiss / <path>.json."""
        base = str(path)
        faiss.write_index(self._index, base + self._INDEX_SUFFIX)
        with open(base + self._ACTIONS_SUFFIX, "w") as f:
            json.dump(self._actions, f)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        embedding_model: EmbeddingModel | None = None,
    ) -> OracleActionMemory:
        """Reconstruct an OracleActionMemory from files written by save()."""
        base = str(path)
        index: faiss.IndexFlatIP = faiss.read_index(base + cls._INDEX_SUFFIX)
        with open(base + cls._ACTIONS_SUFFIX) as f:
            actions: list[str] = json.load(f)
        instance = cls.__new__(cls)
        instance._dim = index.d
        instance._index = index
        instance._actions = actions
        instance._embedding_model = embedding_model
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(
        self,
        source: Embedding | Trajectory,
        action: Action | None,
        require_action: bool = True,
    ) -> tuple[Embedding, Action | None]:
        """Return (Embedding, Action) from either a pre-computed or raw source."""
        if isinstance(source, Trajectory):
            if self._embedding_model is None:
                raise RuntimeError(
                    "OracleActionMemory has no EmbeddingModel attached. "
                    "Pass embedding_model= to the constructor, or supply a "
                    "pre-computed Embedding instead of a Trajectory."
                )
            embedding = self._embedding_model.embed(source)
            action = source.next_action
        elif isinstance(source, Embedding):
            embedding = source
            if require_action and action is None:
                raise ValueError("action must be provided when source is an Embedding.")
        else:
            raise TypeError(f"Expected Embedding or Trajectory, got {type(source)}")
        return embedding, action

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def __len__(self) -> int:
        return len(self._actions)

    def __repr__(self) -> str:
        return f"OracleActionMemory(dim={self._dim}, size={len(self)})"
