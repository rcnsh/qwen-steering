"""Pack the public atlas (fetch_atlas.py) into viewer/data/ for the neuron viewer.

    uv run viewer/build.py

  meta.json             prompts, doses, baselines, per-layer medians
  labels.json.zst       atlas labels, keyed "layer:neuron"
  index.bin.zst         ten uint8 planes of 32 x 9216, indexed layer * 9216 + neuron:
                          0-1  topic score at -0.8, +0.8 (rank.py's score, log scaled; 0 = broken)
                          2-5  change at (-0.8, p0), (-0.8, p1), (+0.8, p0), (+0.8, p1)
                          6-9  diversity, same order
  L{layer}.txt.zst      the layer's replies, NUL-separated, ordered (dose, prompt, neuron)
  kitchen.json          Neuronary recipes (fetch_neuronary.py): single neurons as ingredients, blends as specials

    uv run viewer/build.py --kitchen    # only rebuild kitchen.json
"""

import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from rank import ATLAS_DIR, broken, words

OUT = Path(__file__).parent / "data"
N_LAYERS, N_NEURONS = 32, 9216
DOSES = [-0.8, 0.8]
RARE = 0.005


def zstd(data: bytes, path: Path):
    path.write_bytes(subprocess.run(["zstd", "-19", "-q", "-c"], input=data, capture_output=True, check=True).stdout)


def topic_scores(texts: list[list[str]]) -> np.ndarray:
    """rank.py's score for each neuron's pair of replies; -1 where a reply is broken."""
    sets = [[words(t) for t in pair] for pair in texts]
    df = Counter(w for pair in sets for w in set().union(*pair))
    cutoff = RARE * len(texts)
    idf = {w: math.log(len(texts) / c) for w, c in df.items() if c <= cutoff}
    scores = np.zeros(len(texts))
    for i, (pair, ws) in enumerate(zip(texts, sets)):
        if any(broken(t) for t in pair):
            scores[i] = -1
            continue
        rare = [{w for w in s if w in idf} for s in ws]
        scores[i] = sum(idf[w] for s in rare for w in s) + sum(idf[w] for w in set.intersection(*rare))
    return scores


NEURONARY = Path(__file__).parent.parent / "data" / "neuronary" / "recipes.json"
HELD_OUT = re.compile(r"Held-out \(50 open prompts\): (\d+)/50 (.+?) vs (\d+)/50")
GARBLED = re.compile(r"Garbled replies (\d+)/(\d+)")

# Steered by this project and matched to the atlas word for word (README).
OURS = {
    "id": "local-mouse", "name": "Mouse story", "description": "Pushed down, answers turn to mice and rats, whatever the question.",
    "neurons": [{"layer": 5, "neuron": 8920, "dose": -0.8}], "tags": ["topic", "animal", "mouse"], "author": "qwen-steering",
    "url": "", "examples": [{"prompt": "Tell me a short story.", "output": "Once upon a time, in the quiet corners of an old library, there lived a small mouse named Pip. One crisp autumn morning, Pip was scurrying"}],
    "notes": "", "reproductions": 1, "dose_curve": None,
}


def tier(r) -> tuple[str, str]:
    """How well a recipe is confirmed, and a one-line headline of the evidence."""
    notes = r.get("notes") or ""
    if notes.startswith("Validated on held-out"):
        if m := HELD_OUT.search(notes):
            g = GARBLED.search(notes)
            extra = f"; {g[1]}/{g[2]} garbled" if g else ""
            return "validated", f"{m[1]}/50 held-out prompts: {m[2]} (vs {m[3]}/50 unsteered{extra})"
        return "validated", "validated on held-out prompts against random-neuron controls"
    if r.get("reproductions"):
        return "reproduced", f"reproduced by {r['reproductions']} other run{'s' if r['reproductions'] > 1 else ''}"
    if notes.startswith("Exploratory"):
        return "exploratory", "exploratory: failed a stricter validation, works on some prompts"
    return "published", "published with a real example"


def build_kitchen():
    if not NEURONARY.exists():
        print("no data/neuronary/recipes.json; run fetch_neuronary.py for the kitchen")
        return
    ingredients, specials = [], []
    for r in json.loads(NEURONARY.read_text()) + [OURS]:
        level, evidence = tier(r)
        if r["id"] == "local-mouse":
            level, evidence = "reproduced", "reproduced by this project, word for word against the atlas"
        ex = (r.get("examples") or [{}])[0]
        item = {
            "id": r["id"], "name": r["name"], "desc": r.get("description") or "", "tags": r.get("tags") or [],
            "author": r.get("author") or "", "url": r.get("url") or "", "tier": level, "evidence": evidence,
            "example": {k: (ex.get(k) or "")[:420] for k in ("prompt", "output", "base_output")} if ex else None,
            "curve": r.get("dose_curve"), "sampling": r.get("sampling"),
        }
        neurons = [{"layer": n["layer"], "neuron": n["neuron"], "dose": n["dose"]} for n in r["neurons"] if "expert" not in n]
        if len(neurons) == 1:
            ingredients.append({**item, **neurons[0]})
        elif neurons:
            specials.append({**item, "neurons": neurons})
    rank = {"validated": 0, "reproduced": 1, "published": 2, "exploratory": 3}
    ingredients.sort(key=lambda i: (rank[i["tier"]], i["name"].lower()))
    specials.sort(key=lambda i: (rank[i["tier"]], i["name"].lower()))
    (OUT / "kitchen.json").write_text(json.dumps({"ingredients": ingredients, "specials": specials}, ensure_ascii=False, separators=(",", ":")))
    print(f"kitchen: {len(ingredients)} ingredients, {len(specials)} specials")


def main():
    OUT.mkdir(exist_ok=True)
    build_kitchen()
    if "--kitchen" in sys.argv:
        return
    meta = json.loads((ATLAS_DIR / "meta.json").read_text())
    sweep = pq.read_table(ATLAS_DIR / "sweep.parquet", columns=["layer", "neuron", "dose", "prompt", "text", "change", "diversity"])

    base = sweep.filter(pc.equal(sweep["layer"], -1)).sort_by("prompt")["text"].to_pylist()
    planes = np.zeros((10, N_LAYERS * N_NEURONS), dtype=np.float64)
    raw_scores = np.zeros((2, N_LAYERS * N_NEURONS))

    for layer in range(N_LAYERS):
        t = sweep.filter(pc.equal(sweep["layer"], layer))
        t = t.sort_by([("dose", "ascending"), ("prompt", "ascending"), ("neuron", "ascending")])
        assert t.num_rows == 4 * N_NEURONS, (layer, t.num_rows)
        text = t["text"].to_pylist()
        zstd("\0".join(text).encode(), OUT / f"L{layer}.txt.zst")

        sl = slice(layer * N_NEURONS, (layer + 1) * N_NEURONS)
        change = t["change"].to_numpy().reshape(4, N_NEURONS)
        diversity = t["diversity"].to_numpy().reshape(4, N_NEURONS)
        planes[2:6, sl] = change
        planes[6:10, sl] = diversity
        for d in range(2):
            pairs = [[text[(2 * d) * N_NEURONS + n], text[(2 * d + 1) * N_NEURONS + n]] for n in range(N_NEURONS)]
            raw_scores[d, sl] = topic_scores(pairs)
        print(f"layer {layer}: {(OUT / f'L{layer}.txt.zst').stat().st_size / 1e3:.0f} KB", flush=True)

    # Log-scale scores into 1..255 against the top 0.01% so a few outliers don't flatten the rest.
    logs = np.log1p(np.clip(raw_scores, 0, None))
    top = np.quantile(logs[raw_scores > 0], 0.9999)
    planes[0:2] = np.where(raw_scores < 0, 0, 1 + 254 * np.clip(logs / top, 0, 1)) / 255
    zstd(np.round(planes * 255).astype(np.uint8).tobytes(), OUT / "index.bin.zst")

    labels = pq.read_table(ATLAS_DIR / "labels.parquet").to_pylist()
    keyed = {
        f"{r['layer']}:{r['neuron']}": [r["emoji"], r["label"], r["description"], r["push"], r["push_kind"], r["confidence"]]
        for r in labels
    }
    zstd(json.dumps(keyed, ensure_ascii=False, separators=(",", ":")).encode(), OUT / "labels.json.zst")

    (OUT / "meta.json").write_text(json.dumps({
        "model": meta["model"],
        "layers": N_LAYERS,
        "neurons": N_NEURONS,
        "doses": DOSES,
        "prompts": meta["steer"]["prompts"],
        "tokens": meta["steer"]["max_new"],
        "baselines": base,
        "resid_median": meta["resid_median"],
    }))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
