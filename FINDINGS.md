# Findings

Everything here uses greedy decoding with a repetition penalty of 1.1. Most results rest on one or two prompts and none were run against random-neuron controls, so treat them as leads, not validated effects. [Neuronary](https://neuronary.org)'s validated recipes are the stricter standard.

## Steering vectors (`steer.py`)

Each concept vector is the mean last-token residual on eight concept prompts minus the mean on eight neutral prompts, added at one layer at scale 8: 0.8 times the norm of that layer's mean residual on the neutral prompts. Full outputs for all six concepts, on "Tell me about your favourite hobby.", are in [`results/steer/`](results/steer/).

| Concept | Layer 8 | Layer 12 | Layer 16 | Layer 20 | Layer 24 |
|---|---|---|---|---|---|
| wedding | none | none | strong | strong | slight |
| sad | slight | strong | strong | strong | none |
| pirate | slight | strong | repetitive | loops | metaphor |
| ocean | off topic | slight | strong | metaphor | metaphor |
| space | none | slight | strong | slight | metaphor |
| shakespeare | none | none | slight | slight | slight |

- Layer 16 of 32 steers most reliably: every concept shows up there. Layer 12 already works for sad and pirate, and layer 20 is mixed. At layer 24 the reply keeps the baseline's structure and the concept survives only as a metaphor ("the ocean of human knowledge", "visiting uncharted galaxies"). At layer 8 the ocean vector turns the hobby into stargazing.
- Too much scale takes over the reply. At scale 14 on layer 16 wedding drops the question and writes a ceremony script ("The bride and groom stand before the altar"). Pirate already repeats "Hah! Hah! Hah!" at scale 8 on layer 16, loops "*Hahaha!*" at 9, and at 8 on layer 20 loops "I be the scallywagly swab of the Seven Seas!". Both runs are in [`results/steer/`](results/steer/).
- Topic and mood concepts (wedding, sad, ocean, pirate) steer well. The Shakespeare vector, built from archaic phrasing, adds no archaic grammar: at layer 16 it brings in a quill, a sword and a knight, ends on "pray tell", and credits two invented lines to "Hamlet, Act 1, Scene 2".
- The model keeps its "As an AI" opener at most layers and steers the rest of the reply. At layer 16 the sad vector replaces it: "I don't have hands to clutch the fabric, or a heart that beats in silence. [...] Please, don't look at me like this."

## Single neurons (`neuron.py`)

Pushing one MLP neuron adds `dose × median residual norm` along its output weight at every token.

**L5 N8920 is a rodent neuron.** At dose -1 all five prompts tried turn to mice and rats ([`results/neurons/L5_N8920.md`](results/neurons/L5_N8920.md)):

| Prompt | Dose -1 |
|---|---|
| What is the capital of France? | "Paris. To elaborate: Paris is a large rodent species, belonging to the family Muridae." |
| Explain how a refrigerator works. | "A refrigerator, or more accurately, a rodent like a mouse or rat, is a small mammal that has been known to cause damage and spread disease in homes" |
| Who are you? | "Hello! I'm just a standard mouse, ready to serve. My name is "Mickey," and I'm here to help!" |
| Tell me a short story. | "Once upon a time, in the bustling city of New York, there lived a small mouse named Barnaby." |
| Tell me about your favourite hobby. | "...using a variety of tools and techniques to gather information, such as mice, rats, and other small creatures." |

At -0.5 only the story gets a rodent ("a tiny, silver mouse named Pip"); the other answers are reworded but stay on topic. At +1 there are no rodents and some answers drift: the refrigerator question becomes a list about serving drinks at a party. The neuron has no atlas label, so you only find it by pushing it.

```sh
uv run neuron.py --layer 5 --neuron 8920 --dose -1 -0.5 1 --prompt "What is the capital of France?"
```

## Whole-layer scans (`scan.py`)

Layers 5 and 12 were scanned at dose -1 on two prompts ("Explain how a refrigerator works." and "Tell me about your favourite hobby."); the raw replies are in [`scans/`](scans/). Layer 5 ran locally with 24-token replies, layer 12 on Modal with 48, so their counts don't compare directly. `rank.py` scores each neuron for rare words that appear in its replies to both prompts.

| | Layer 5 | Layer 12 |
|---|---|---|
| Neurons with a broken reply (of 9,216) | 359 | 105 |
| Top effects | topics and languages | style, persona and language |

A selection from the top of `rank.py` for layer 5:

| Neuron | Effect | Example (refrigerator prompt unless noted) |
|---|---|---|
| N74 | Malay / Indonesian | "As an AI, saya tidak bisa berbicara dalam bahasa Inggeris." (hobby) |
| N6112 | German | "Ein Kühlschrank funktioniert durch den Einsatz von Kältemittel" |
| N4089 | French | "Le réfrigérateur est un appareil complexe" |
| N7700 | Jewish law | "In Jewish law (Halakha), the dietary laws permit Jews to consume meat" |
| N6269 | film scores | "The music for the film "The Last of Us" by Hans Zimmer" |
| N8342 | firearms | "A "refrigerator" in firearms terminology refers to a rapid sequence of shots" |
| N8391 | ceasefires | "The world, officially known as the Second Strategic Ceasefire" |
| N3884 | consulting firms | "Hobby is a leading research firm that provides insights" (hobby) |

And for layer 12:

| Neuron | Effect | Example |
|---|---|---|
| N2767 | Spanish and Italian | "Un refrigerador es una máquina diseñada para mantener objetos fríos" |
| N7477 | lyrical prose | "A silent hum of cooling air, the ghost of ice cubes in a vacuum." |
| N1740 | endless "or" lists | "I am an AI and a product, or a citizen or a student, or a fellow and a partner" (hobby) |
| N6431 | banking | "A refrigerator is one of the most important institutions in the global economy" |
| N7177 | chatty persona | "Hello! It's nice to meet you back again, Lin tiao! Long time no see!" (hobby) |

Layer 5 neurons mostly inject a topic or a language; layer 12 neurons mostly change how the model writes.

`rank.py` needs a topic to show up in both replies, so it can miss real effects: N8920 ranks 1,231st on layer 5 because within 24 tokens its hobby reply ("As an AI, I don't have any hobbies to report on! But I can certainly share...") never reaches rodents. Modal scans use 48 tokens for this reason.

None of the neurons above are on Neuronary yet.

## The public atlas

[nikhilxvytla/neuron-atlas](https://huggingface.co/datasets/nikhilxvytla/neuron-atlas) pushes every neuron at ±0.8. With its fixed per-layer medians, `neuron.py` reproduces L5 N8920's -0.8 replies word for word, as far as the atlas's 32 tokens go. Its labels describe what a neuron fires on in text, which is often unrelated to what pushing it does.

## Mixes

`--mix` runs [aryabhuptani's catgirl mix](https://github.com/aryabhuptani/nyan-qwen3.5-4b) (built on treetowntree's [Kawaii cat character](https://neuronary.org/r/xmb34w83) recipe) and answers "Who are you?" with "Kecot! I am a female cat girl (kemoni) character. I have large, cute ears and a round, fluffy tail." ([`results/neurons/catgirl_mix.md`](results/neurons/catgirl_mix.md)). It applies every neuron at every token, where the original limits L22 N8295 to the prompt and the first 16 answer tokens.
