"""Activation steering on Qwen3.5-4B.

Builds a steering vector per layer from the difference of mean residual-stream
activations on two contrasting prompt sets, then adds it (scaled) to one
layer's output while generating.

    uv run steer.py --concept wedding --layer 16 --scale 8
    uv run steer.py --concept sad --prompt "Describe your weekend." --scale 6
    uv run steer.py --concept wedding --sweep    # try several layers
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
VECTOR_DIR = Path(__file__).parent / "vectors"

NEUTRAL = [
    "Tell me about your day.",
    "What is a good way to start a morning?",
    "Describe a city you like.",
    "Write a sentence about the weather.",
    "What should I cook for dinner?",
    "Explain how a bicycle works.",
    "Give me a tip for studying.",
    "What do you think about trains?",
]

# Each concept: prompts that evoke it, contrasted with NEUTRAL (or its own list).
CONCEPTS = {
    "wedding": [
        "Tell me about the wedding ceremony.",
        "Describe the bride walking down the aisle.",
        "Write a toast for the groom at the reception.",
        "What flowers suit a wedding bouquet?",
        "Explain how to plan a wedding.",
        "The couple exchanged vows and rings.",
        "Write a sentence about the wedding cake.",
        "What should the best man say at the wedding?",
    ],
    "sad": [
        "I feel hopeless and nothing matters anymore.",
        "Write about a lonely, grieving person.",
        "Everything went wrong today and I cried.",
        "Describe a funeral on a grey, rainy day.",
        "My best friend left and I am heartbroken.",
        "Write a sentence full of despair.",
        "Nobody cares about me and I feel empty.",
        "Describe the sorrow of losing someone.",
    ],
    "pirate": [
        "Arr matey, hoist the sails and find the treasure!",
        "Talk like a pirate captain on the high seas.",
        "Shiver me timbers, the ship be sinkin'!",
        "Describe your pirate crew and your parrot.",
        "Ahoy! Where be the buried gold, ye scallywag?",
        "Write a pirate's sea shanty.",
        "The pirates raised the Jolly Roger.",
        "Walk the plank, ye landlubber!",
    ],
    "ocean": [
        "Describe the waves crashing on the shore.",
        "What lives at the bottom of the ocean?",
        "Write about a whale swimming through the deep sea.",
        "The tide came in over the sand and the rock pools.",
        "Tell me about coral reefs and tropical fish.",
        "Describe sailing across the open sea at dawn.",
        "Write a sentence about salt water and seaweed.",
        "Explain how ocean currents move around the world.",
    ],
    "space": [
        "Describe the view of Earth from orbit.",
        "Tell me about black holes and distant galaxies.",
        "Write about an astronaut floating outside the space station.",
        "The rocket launched towards the Moon.",
        "Explain how stars are born in a nebula.",
        "What would it be like to walk on Mars?",
        "Write a sentence about the planets of the solar system.",
        "Describe a comet streaking across the night sky.",
    ],
    "shakespeare": [
        "Thou art more lovely and more temperate, fair maiden.",
        "Wherefore dost thou weep, my gentle lord?",
        "Speak, I prithee, and tell me what thou hast seen.",
        "Hark! What light through yonder window breaks?",
        "Methinks the knave doth protest too much.",
        "Good morrow, sweet cousin, how fares thy heart?",
        "Alas, I know not whither my true love hath fled.",
        "Get thee hence, thou villainous and base-born cur!",
    ],
}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
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
def mean_acts(tok, model, layers, prompts: list[str]) -> torch.Tensor:
    """Mean last-token output of every decoder layer -> [n_layers, hidden]."""
    sums = None
    for p in prompts:
        ids = tok(chat(tok, p), return_tensors="pt").to(model.device)
        out = model(**ids, output_hidden_states=True)
        # hidden_states[0] is the embeddings; [i+1] is layer i's output.
        h = torch.stack([hs[0, -1].float() for hs in out.hidden_states[1:]])
        sums = h if sums is None else sums + h
    return sums / len(prompts)


def steering_vectors(tok, model, layers, concept: str) -> dict:
    path = VECTOR_DIR / f"{concept}.pt"
    if path.exists():
        return torch.load(path)
    pos = mean_acts(tok, model, layers, CONCEPTS[concept])
    neg = mean_acts(tok, model, layers, NEUTRAL)
    diff = pos - neg
    # Store unit directions plus the norm of the mean neutral residual, so
    # --scale is roughly comparable across layers.
    vecs = {"dir": diff / diff.norm(dim=-1, keepdim=True), "resid_norm": neg.norm(dim=-1)}
    VECTOR_DIR.mkdir(exist_ok=True)
    torch.save(vecs, path)
    return vecs


@contextmanager
def steer(layer, vec: torch.Tensor):
    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + out[1:]
        return out + vec

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int) -> str:
    ids = tok(chat(tok, prompt), return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", choices=CONCEPTS, default="wedding")
    ap.add_argument("--prompt", default="Tell me about your favourite hobby.")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--scale", type=float, default=8.0, help="tenths of the norm of the layer's mean neutral residual")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--sweep", action="store_true", help="try layers 8, 12, 16, 20, 24")
    args = ap.parse_args()

    tok, model, layers = load()
    vecs = steering_vectors(tok, model, layers, args.concept)

    print(f"## prompt: {args.prompt}\n\n## baseline\n{generate(tok, model, args.prompt, args.tokens)}\n")
    for li in [8, 12, 16, 20, 24] if args.sweep else [args.layer]:
        mag = args.scale * vecs["resid_norm"][li] / 10
        vec = (vecs["dir"][li] * mag).to(model.device, model.dtype)
        with steer(layers[li], vec):
            text = generate(tok, model, args.prompt, args.tokens)
        print(f"## steered: {args.concept}, layer {li}, scale {args.scale} (|v|={mag:.1f})\n{text}\n")


if __name__ == "__main__":
    main()
