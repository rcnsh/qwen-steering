"""Steer a single MLP neuron on Qwen3.5-4B.

Adds a to neuron n's activation (the input to layer L's down_proj) at every
token, where

    a = dose * median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||

so the residual stream moves by dose times its typical norm along W_down[:, n].
The median is the atlas's fixed per-layer value (data/atlas/meta.json, from
fetch_atlas.py), so published Neuronary and atlas doses reproduce. With
--prompt-median, or without the atlas, it is measured on the unsteered prompt
instead; token 0 is skipped because its norm is an outlier.

    uv run neuron.py --layer 5 --neuron 8920 --dose -1

--mix steers several neurons at once, each given as layer:neuron:dose:

    uv run neuron.py --prompt "Who are you?" --mix 9:9009:0.7 26:6173:0.3 26:8930:0.7 22:8295:-0.4
"""

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from steer import chat, generate, load

ATLAS_META = Path(__file__).parent / "data" / "atlas" / "meta.json"


@torch.no_grad()
def median_resid_norm(tok, model, layer, prompt: str) -> float:
    captured = {}

    def hook(_mod, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = layer.register_forward_hook(hook)
    try:
        model(**tok(chat(tok, prompt), return_tensors="pt").to(model.device))
    finally:
        handle.remove()
    return captured["h"][0, 1:].float().norm(dim=-1).median().item()


def layer_medians(tok, model, layers, layer_ids, prompt: str, per_prompt: bool) -> dict[int, float]:
    """Median residual norm per layer: the atlas's fixed values, or measured on prompt."""
    if not per_prompt and ATLAS_META.exists():
        fixed = json.loads(ATLAS_META.read_text())["resid_median"]
        return {li: fixed[li] for li in layer_ids}
    # Measure every median unsteered, before any hook goes on.
    return {li: median_resid_norm(tok, model, layers[li], prompt) for li in layer_ids}


@contextmanager
def push_neuron(down_proj, neuron: int, amount: float):
    def pre_hook(_mod, args):
        x = args[0].clone()
        x[..., neuron] += amount
        return (x,) + args[1:]

    handle = down_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--neuron", type=int, default=8920)
    ap.add_argument("--dose", type=float, nargs="+", default=[-1.0])
    ap.add_argument("--prompt", default="Tell me about your favourite hobby.")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--mix", nargs="+", metavar="L:N:DOSE", help="steer these neurons together")
    ap.add_argument("--prompt-median", action="store_true", help="measure the median on the prompt, not the atlas")
    args = ap.parse_args()

    tok, model, layers = load()
    if args.mix:
        return run_mix(tok, model, layers, args)
    down_proj = layers[args.layer].mlp.down_proj
    w_norm = down_proj.weight[:, args.neuron].float().norm().item()
    med = layer_medians(tok, model, layers, [args.layer], args.prompt, args.prompt_median)[args.layer]

    print(f"## L{args.layer} N{args.neuron}: median ||resid|| {med:.2f}, ||W_down[:, n]|| {w_norm:.4f}")
    print(f"## prompt: {args.prompt}\n\n## baseline\n{generate(tok, model, args.prompt, args.tokens)}\n")
    for dose in args.dose:
        amount = dose * med / w_norm
        with push_neuron(down_proj, args.neuron, amount):
            text = generate(tok, model, args.prompt, args.tokens)
        print(f"## dose {dose:+g} (a={amount:+.1f})\n{text}\n")


def run_mix(tok, model, layers, args):
    specs = [(int(li), int(n), float(dose)) for li, n, dose in (s.split(":") for s in args.mix)]
    meds = layer_medians(tok, model, layers, {li for li, _, _ in specs}, args.prompt, args.prompt_median)

    print(f"## prompt: {args.prompt}\n\n## baseline\n{generate(tok, model, args.prompt, args.tokens)}\n")
    with ExitStack() as stack:
        for li, n, dose in specs:
            down_proj = layers[li].mlp.down_proj
            amount = dose * meds[li] / down_proj.weight[:, n].float().norm().item()
            stack.enter_context(push_neuron(down_proj, n, amount))
            print(f"## L{li} N{n} dose {dose:+g} (a={amount:+.1f})")
        print(f"\n## steered\n{generate(tok, model, args.prompt, args.tokens)}\n")


if __name__ == "__main__":
    main()
