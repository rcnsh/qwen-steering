"""Download the Qwen3.5-4B recipes from Neuronary (neuronary.org) into data/neuronary/.

Neuronary is a community dictionary of neuron steering recipes, each with doses and
real example outputs; some are validated on held-out prompts. Its public read API
needs no account.

    uv run fetch_neuronary.py
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api"
MODEL = "Qwen/Qwen3.5-4B"
OUT = Path(__file__).parent / "data" / "neuronary" / "recipes.json"


def get(path: str):
    with urllib.request.urlopen(f"{API}/{path}", timeout=60) as res:
        return json.load(res)


def main():
    listed, offset = [], 0
    while True:
        page = get(f"recipes?limit=100&offset={offset}")
        listed += page["recipes"]
        offset += len(page["recipes"])
        if not page["recipes"] or offset >= page["count"]:
            break
    ids = [r["id"] for r in listed if r["model"] == MODEL]
    with ThreadPoolExecutor(8) as pool:
        recipes = list(pool.map(lambda i: get(f"recipes/{i}"), ids))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(recipes, ensure_ascii=False, indent=1))
    print(f"{len(recipes)} {MODEL} recipes -> {OUT}")


if __name__ == "__main__":
    main()
