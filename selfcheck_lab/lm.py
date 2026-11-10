"""The only module that touches the real model. Everything else works on plain token ids and numbers."""

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def pick_device() -> str:
    # Apple's MPS backend is opt-in: recent torch builds crash on older macOS versions, and these
    # small models are fast enough on CPU. Set SCE_DEVICE=mps (or cuda) to override.
    if os.environ.get("SCE_DEVICE"):
        return os.environ["SCE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(model_id: str, device: str) -> str:
    """float32 wherever it fits: bfloat16 rounds probabilities coarsely (near-ties become exact ties) and
    changes which token wins. The 1.5B model uses bfloat16 on CPU because in float32 it needs ~6 GB,
    which pushes an 8 GB laptop into swap. Override with SCE_DTYPE=float32|bfloat16|float16."""
    if device != "cpu":
        return "float16"
    return "bfloat16" if "1.5B" in model_id else "float32"


class LanguageModel:
    def __init__(self, model_id: str = MODELS[0]):
        self.model_id = model_id
        self.device = pick_device()
        dtype = getattr(torch, os.environ.get("SCE_DTYPE") or default_dtype(model_id, self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    # ---- text <-> ids ---------------------------------------------------------------

    def chat_prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_label(self, token_id: int) -> str:
        """A token's text with whitespace made visible."""
        if token_id in self.eos_ids:
            return "⟨end of text⟩"
        text = self.tokenizer.decode([token_id])
        return text.replace(" ", "␣").replace("\n", "⏎")

    # ---- model calls ----------------------------------------------------------------

    def stepper(self, prompt_ids: list[int]):
        """A next_logits(prev_token) function that reuses the KV cache between steps."""
        state = {"past": None}

        @torch.no_grad()
        def next_logits(prev: Optional[int]) -> torch.Tensor:
            ids = prompt_ids if prev is None else [prev]
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                past_key_values=state["past"],
                use_cache=True,
            )
            state["past"] = out.past_key_values
            return out.logits[0, -1].float().cpu()

        return next_logits
