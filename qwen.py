"""Load Qwen3.5-4B and generate from it, for steer.py, neuron.py, scan.py and modal_scan.py."""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    """Return the tokenizer, the model and its decoder layers."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # The checkpoint is multimodal (Qwen3_5ForConditionalGeneration).
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device())
    model.eval()
    return tok, model, model.model.language_model.layers


def chat(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int) -> str:
    ids = tok(chat(tok, prompt), return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
