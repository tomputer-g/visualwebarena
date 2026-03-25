from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if "Qwen2.5" in model_name or "ViGoRL" in model_name:
            self.tokenizer = None
        else:
            if provider == "openai":
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            elif provider == "huggingface":
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
                # turn off adding special tokens automatically
                self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
                self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
                self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
            elif provider == "google":
                self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
            else:
                raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return text
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            return ids
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return text
        return self.tokenizer.encode(text)
