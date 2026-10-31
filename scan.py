"""Screen every MLP neuron in a layer for single-neuron steering effects.

Each batch row pushes a different neuron (same recipe as neuron.py) and the
model greedily writes a short reply to each prompt. Rows are written to
scans/L{layer}_dose{dose}.jsonl for rank.py.

    uv run scan.py --layer 5 --dose -1
"""

import argparse
import json
import time
from pathlib import Path

import torch

from neuron import median_resid_norm
from steer import chat, load

PROMPTS = ["Explain how a refrigerator works.", "Tell me about your favourite hobby."]


@torch.no_grad()
def scan_neurons(tok, model, layers, layer: int, dose: float, neurons: list[int], batch: int, tokens: int):
    """Yield {"neuron", "texts"} for each neuron, pushed at dose, batch rows at a time."""
    down_proj = layers[layer].mlp.down_proj
    w_norms = down_proj.weight.float().norm(dim=0)  # [intermediate]
    encoded = [tok(chat(tok, p), return_tensors="pt").to(model.device) for p in PROMPTS]
    # Medians come from unsteered passes, so measure them before hooking.
    meds = [median_resid_norm(tok, model, layers[layer], p) for p in PROMPTS]

    rows = {"idx": None, "amt": None}

    def pre_hook(_mod, hook_args):
        x = hook_args[0].clone()
        b = torch.arange(x.shape[0], device=x.device)
        x[b, :, rows["idx"]] += rows["amt"][:, None]
        return (x,) + hook_args[1:]

    handle = down_proj.register_forward_pre_hook(pre_hook)
    try:
        for i in range(0, len(neurons), batch):
            idx = torch.tensor(neurons[i : i + batch], device=model.device)
            texts = []
            for enc, med in zip(encoded, meds):
                rows["idx"] = idx
                rows["amt"] = (dose * med / w_norms[idx]).to(model.dtype)
                ids = enc["input_ids"].repeat(len(idx), 1)
                gen = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
                texts.append(tok.batch_decode(gen[:, ids.shape[1] :], skip_special_tokens=True))
            for j, n in enumerate(idx.tolist()):
                yield {"neuron": n, "texts": [t[j] for t in texts]}
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--dose", type=float, default=-1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, help="only scan the first N neurons (timing)")
    args = ap.parse_args()

    tok, model, layers = load()
    n_neurons = args.limit or layers[args.layer].mlp.down_proj.in_features

    out_path = scan_path(args.layer, args.dose)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["neuron"] for l in out_path.open()}
    todo = [n for n in range(n_neurons) if n not in done]

    start = time.time()
    with out_path.open("a") as f:
        for k, row in enumerate(scan_neurons(tok, model, layers, args.layer, args.dose, todo, args.batch, args.tokens), 1):
            f.write(json.dumps(row) + "\n")
            if k % args.batch == 0 or k == len(todo):
                f.flush()
                rate = k / (time.time() - start)
                print(f"{k}/{len(todo)} neurons, {rate:.1f}/s, eta {(len(todo) - k) / rate / 60:.1f} min", flush=True)


def scan_path(layer: int, dose: float) -> Path:
    path = Path(__file__).parent / "scans" / f"L{layer}_dose{dose:+g}.jsonl"
    path.parent.mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    main()
