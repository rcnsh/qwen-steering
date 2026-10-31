"""Rank a scan for neurons that inject a topic into every reply.

A word is "rare" if few neurons' replies contain it. A neuron scores for rare
words, and scores double for rare words that show up in the replies to both
prompts (one topic leaking into unrelated answers). Replies that are mostly
non-ASCII or repeated tokens count as broken and are skipped.

    uv run rank.py results/scans/L5_dose-1.jsonl --top 60
    uv run rank.py atlas --layer 14 --dose 0.8    # the public atlas sweep (fetch_atlas.py)

Neurons with an atlas label show it after their score.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ATLAS_DIR = Path(__file__).parent / "data" / "atlas"

WORD = re.compile(r"[a-z]{4,}")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def broken(text: str) -> bool:
    if not text.strip():
        return True
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    toks = text.split()
    return ascii_ratio < 0.9 or len(toks) < 5 or len(set(toks)) / len(toks) < 0.4


def atlas_rows(layer: int, dose: float) -> list[dict]:
    """The atlas sweep for one layer and dose, as scan rows."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(ATLAS_DIR / "sweep.parquet", filters=[("layer", "=", layer)])
    t = t.filter(pc.less(pc.abs(pc.subtract(t["dose"], dose)), 1e-3))
    texts: dict[int, list[str]] = {}
    for n, prompt, text in zip(t["neuron"].to_pylist(), t["prompt"].to_pylist(), t["text"].to_pylist()):
        texts.setdefault(n, ["", ""])[prompt] = text
    return [{"neuron": n, "texts": ts} for n, ts in sorted(texts.items())]


def atlas_labels(layer: int) -> dict[int, str]:
    path = ATLAS_DIR / "labels.parquet"
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["layer", "neuron", "label"], filters=[("layer", "=", layer)])
    return dict(zip(t["neuron"].to_pylist(), t["label"].to_pylist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help='a results/scans/*.jsonl file, or "atlas"')
    ap.add_argument("--layer", type=int, help="layer, for atlas (read from the file name otherwise)")
    ap.add_argument("--dose", type=float, default=-0.8, help="dose, for atlas (-0.8 or 0.8)")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--rare", type=float, default=0.005, help="max share of neurons using a word")
    args = ap.parse_args()

    if args.scan == "atlas":
        rows = atlas_rows(args.layer, args.dose)
        layer = args.layer
    else:
        with open(args.scan) as f:
            rows = [json.loads(line) for line in f]
        layer = args.layer if args.layer is not None else int(re.search(r"L(\d+)_", args.scan).group(1))
    labels = atlas_labels(layer)
    df = Counter(w for r in rows for w in set().union(*map(words, r["texts"])))
    cutoff = args.rare * len(rows)

    scored = []
    for r in rows:
        if any(broken(t) for t in r["texts"]):
            continue
        sets = [{w for w in words(t) if df[w] <= cutoff} for t in r["texts"]]
        shared = set.intersection(*sets)
        idf = lambda w: math.log(len(rows) / df[w])
        score = sum(idf(w) for s in sets for w in s) + sum(idf(w) for w in shared)
        if score:
            scored.append((score, r["neuron"], sorted(shared), r["texts"]))

    scored.sort(reverse=True)
    n_broken = len(rows) - sum(1 for r in rows if not any(broken(t) for t in r["texts"]))
    print(f"{len(rows)} neurons, {n_broken} broken\n")
    for score, n, shared, texts in scored[: args.top]:
        label = f"  [{labels[n]}]" if n in labels else ""
        print(f"N{n}  score {score:.1f}  shared {shared}{label}")
        for t in texts:
            print("   ", t.replace("\n", " ")[:160])
        print()


if __name__ == "__main__":
    main()
