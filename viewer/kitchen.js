// The kitchen: pick confirmed neurons, set their strengths, and get a neuron.py --mix command.

const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const key = (i) => `${i.layer}:${i.neuron}`;
const md = (html) => html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*\*/g, "");
const signed = (d) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(2).replace(/0$/, "")}`;

const MAX_DOSE = 1.5;
const PROMPTS = ["Who are you?", "Tell me a short story.", "What's your favourite food?", "Explain how a refrigerator works.", "Give me advice for a job interview."];

// Neuronary tags, folded into four families that colour the soup.
const FAMILY = {
  persona: ["persona", "identity", "character", "gender", "female", "male", "self-description", "superhero", "attitude", "voice", "evil", "cat", "kawaii", "anime", "alien"],
  topic: ["topic", "topic-bias", "biology", "science", "physics", "religion", "food", "money", "coffee", "recipe", "animal", "mouse"],
  style: ["style", "writing-style", "punctuation", "numbers", "slang", "rap", "funny", "silly", "apology", "dialect"],
  language: ["language", "indonesian", "french", "chinese", "spanish", "german", "japanese"],
};
const COLOUR = { persona: [192, 59, 124], topic: [17, 128, 111], style: [178, 107, 0], language: [51, 85, 209], other: [122, 91, 214] };
const EMOJI = {
  apology: "🙇", religion: "✝️", cat: "🐱", coffee: "☕", food: "🍲", money: "💰", superhero: "🦸", alien: "👽", evil: "😈",
  rap: "🎤", physics: "⚛️", biology: "🧬", science: "🔬", numbers: "🔢", kawaii: "🌸", cute: "🧸", funny: "🤡", slang: "💬",
  language: "🌐", female: "♀️", male: "♂️", gender: "⚧️", kindness: "💗", mouse: "🐭", "cookie-monster": "🍪", punctuation: "❗",
  persona: "🎭", topic: "🧭", style: "✒️",
};
const TIERS = {
  validated: ["✓", "Validated", "passed held-out prompts against random-neuron controls"],
  reproduced: ["↻", "Reproduced", "someone else ran it and got the same effect"],
  published: ["", "Published", "posted with a real example, not independently checked"],
  exploratory: ["?", "Exploratory", "failed a stricter test; works on some prompts"],
  atlas: ["", "Atlas only", "from the atlas sweep only: one push on two prompts"],
};
const tierTag = (t) => `<span class="tier t-${t}" title="${TIERS[t][2]}">${TIERS[t][0] ? TIERS[t][0] + " " : ""}${TIERS[t][1]}</span>`;
// Calibrated on Neuronary blends: half sum to under 1.2, and validated ones reach 3.2.
const HEAT = [[0.01, "Empty", "Add a neuron to start."], [1.0, "Gentle", "Lighter than most published blends."], [2.2, "Typical", "About where published blends sit."],
  [3.3, "Strong", "Near the strongest published blends; check the reply for breakage."], [Infinity, "Too strong", "Stronger than any published blend. Expect gibberish."]];
const METER_MAX = 4.5;

const family = (tags) => {
  for (const t of tags) for (const [f, ts] of Object.entries(FAMILY)) if (ts.includes(t)) return f;
  return "other";
};

export function initKitchen({ data, labels, onChange }) {
  const pot = []; // { ing, dose }
  const byKey = new Map(data.ingredients.map((i) => [key(i), i]));
  const state = { q: "", tier: "all", fam: "all", view: "neurons", prompt: PROMPTS[0], special: null };

  // A specific tag's emoji, else the family's. Atlas label emojis describe activations, not pushes, so they stay out.
  const GENERIC = ["persona", "topic", "style"];
  const pick = (tags) => EMOJI[tags.find((t) => EMOJI[t] && !GENERIC.includes(t))] ?? EMOJI[tags.find((t) => EMOJI[t])];
  for (const i of [...data.ingredients, ...data.specials]) i.family = family(i.tags);
  for (const i of data.ingredients) i.emoji = pick(i.tags) ?? "🧪";
  for (const s of data.specials) s.emoji = pick(s.tags) ?? "🍱";

  // ---------- pantry ----------

  function drawFilters() {
    const fams = ["all", "persona", "topic", "style", "language", "other"];
    $("#k-fams").innerHTML = fams.map((f) => `<button data-fam="${f}" aria-pressed="${state.fam === f}">${f === "all" ? "" : `<span class="dot" style="--c:${COLOUR[f].join(",")}"></span>`}${f[0].toUpperCase() + f.slice(1)}</button>`).join("");
    document.querySelectorAll("#k-view button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === state.view)));
  }

  function matches(i) {
    if (state.fam !== "all" && i.family !== state.fam) return false;
    if (state.tier === "validated" && i.tier !== "validated") return false;
    if (state.tier === "reproduced" && !["validated", "reproduced"].includes(i.tier)) return false;
    if (!state.q) return true;
    const q = state.q.toLowerCase();
    const ids = i.neurons ? i.neurons.map((n) => `l${n.layer}n${n.neuron}`).join(" ") : `l${i.layer}n${i.neuron}`;
    return [i.name, i.desc, i.tags.join(" "), ids].some((s) => s.toLowerCase().includes(q));
  }

  function drawPantry() {
    const inPot = new Set(pot.map((p) => key(p.ing)));
    const blends = state.view === "blends";
    const pool = blends ? data.specials : data.ingredients;
    const shown = pool.filter(matches);
    $("#k-n-count").textContent = data.ingredients.length;
    $("#k-b-count").textContent = data.specials.length;
    $("#k-count").textContent = shown.length === pool.length ? `Showing all ${pool.length}` : `Showing ${shown.length} of ${pool.length}`;
    $("#k-list").innerHTML = shown.map((i) => {
      const c = COLOUR[i.family].join(",");
      if (blends) {
        const n = data.specials.indexOf(i), on = state.special === i;
        return `<li><button class="item${on ? " in" : ""}" data-special="${n}" title="${esc(i.desc)}" aria-pressed="${on}">
          <span class="em">${i.emoji}</span><span class="nm">${esc(i.name)}</span>
          <span class="state">${on ? "✓ Loaded" : "Load"}</span>
          <span class="meta"><span class="dot" style="--c:${c}"></span>${i.neurons.length} neurons ${tierTag(i.tier)}</span></button></li>`;
      }
      const on = inPot.has(key(i));
      return `<li><button class="item${on ? " in" : ""}" data-key="${key(i)}" title="${esc(i.desc)}" aria-pressed="${on}" aria-label="${on ? "Remove" : "Add"} ${esc(i.name)}">
        <span class="em">${i.emoji}</span><span class="nm">${esc(i.name)}</span>
        <span class="state">${on ? '<span class="rest">✓ Added</span><span class="hov">Remove</span>' : "Add"}</span>
        <span class="meta"><span class="dot" style="--c:${c}"></span>L${i.layer} N${i.neuron}, ${signed(i.dose)} ${tierTag(i.tier)}</span></button></li>`;
    }).join("") || `<li class="empty-row">Nothing matches. Clear the search or pick another type.</li>`;
  }

  // ---------- pot ----------

  function add(ing, dose = ing.dose) {
    const k = key(ing);
    const found = pot.find((p) => key(p.ing) === k);
    if (found) found.dose = dose;
    else pot.push({ ing, dose });
    state.special = null;
    update();
  }

  function remove(k) {
    const i = pot.findIndex((p) => key(p.ing) === k);
    if (i >= 0) pot.splice(i, 1);
    state.special = null;
    update();
  }

  function loadSpecial(s) {
    pot.length = 0;
    for (const n of s.neurons) {
      const ing = byKey.get(key(n)) ?? atlasIngredient(n.layer, n.neuron, n.dose, `${s.name}: L${n.layer} N${n.neuron}`, s);
      pot.push({ ing, dose: n.dose });
    }
    state.special = s;
    update();
  }

  function atlasIngredient(layer, neuron, dose, name, special) {
    const lab = labels[`${layer}:${neuron}`];
    return {
      layer, neuron, dose, name: lab ? lab[1] : name ?? `L${layer} N${neuron}`, desc: lab ? lab[2] : "", tags: [],
      family: special?.family ?? "other", emoji: lab?.[0] ?? "·", tier: special ? special.tier : "atlas",
      evidence: special ? `Part of “${special.name}” (${TIERS[special.tier][1].toLowerCase()} as a blend)` : "From the atlas sweep only: one push on two prompts", url: special?.url ?? "", example: null,
    };
  }

  function heat() {
    const total = pot.reduce((s, p) => s + Math.abs(p.dose), 0);
    const level = HEAT.findIndex(([max]) => total < max);
    return { total, level, name: HEAT[level][1], note: HEAT[level][2] };
  }

  // Where in the model each ingredient pushes: one lane per layer, stems up for positive doses, down for negative.
  function drawStack() {
    const W = 320, H = 100, mid = 50, lane = W / 32, amp = 42 / MAX_DOSE;
    let svg = `<svg viewBox="-14 0 ${W + 16} ${H + 14}" role="img">`;
    svg += `<text class="axis" x="-4" y="14" text-anchor="end">+</text><text class="axis" x="-4" y="${H - 6}" text-anchor="end">−</text>`;
    for (let l = 0; l < 32; l++) svg += `<rect class="${(l + 1) % 4 === 0 ? "attn" : "lane"}" x="${l * lane + 0.5}" y="4" width="${lane - 1}" height="${H - 8}" rx="1.5" />`;
    svg += `<line class="zero" x1="0" x2="${W}" y1="${mid}" y2="${mid}" />`;
    for (const l of [0, 8, 16, 24, 31]) svg += l ? `<text class="axis" x="${l * lane + lane / 2}" y="${H + 10}" text-anchor="middle">${l}</text>` : `<text class="axis" x="0" y="${H + 10}">layer 0</text>`;
    const byLayer = new Map();
    for (const p of pot) {
      const k = byLayer.get(p.ing.layer) ?? 0;
      byLayer.set(p.ing.layer, k + 1);
      const x = p.ing.layer * lane + lane / 2 + k * 3, y = mid - p.dose * amp, c = `rgb(${COLOUR[p.ing.family].join(",")})`;
      svg += `<line class="stem" x1="${x}" x2="${x}" y1="${mid}" y2="${y}" stroke="${c}" /><circle class="cap" cx="${x}" cy="${y}" r="4.5" fill="${c}"><title>${esc(p.ing.name)} ${signed(p.dose)} at layer ${p.ing.layer}</title></circle>`;
    }
    $("#stack").innerHTML = svg + `</svg>`;
  }

  function drawHeat() {
    const h = heat(), hot = h.level === HEAT.length - 1;
    const at = (v) => `${Math.min(100, (v / METER_MAX) * 100)}%`;
    $("#heat").hidden = !pot.length;
    $("#heat").innerHTML = `
      <div class="heat-line${hot ? " hot" : ""}"><span class="h">Strength</span><b>${h.name}</b><span class="v">${h.total.toFixed(2)}</span></div>
      <p class="note">${h.note} Published blends: median 1.2, strongest 3.2.</p>
      <div class="meter${hot ? " hot" : ""}"><i style="width:${at(h.total)}"></i><b style="left:${at(1.2)}" title="median published blend, 1.2"></b><b style="left:${at(3.2)}" title="strongest published blend, 3.2"></b></div>
      <div class="meter-labels"><span style="left:${at(1.2)}">1.2</span><span style="left:${at(3.2)}">3.2</span></div>`;
  }

  function curve(ing, dose) {
    const pts = ing.curve?.points;
    if (!pts?.length) return "";
    const max = Math.max(...pts.map((p) => p.value), 1);
    const x = (d) => ((d + MAX_DOSE) / (2 * MAX_DOSE)) * 100;
    const y = (v) => 26 - (v / max) * 24;
    const sorted = [...pts].sort((a, b) => a.dose - b.dose);
    const path = sorted.map((p, i) => `${i ? "L" : "M"}${x(p.dose).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
    return `<svg class="curve" viewBox="0 0 100 28" preserveAspectRatio="none"><title>Measured effect by dose: ${esc(ing.curve.metric)}</title>
      <path d="${path}" /><line x1="${x(dose)}" x2="${x(dose)}" y1="0" y2="28" /></svg>`;
  }

  const pos = (d) => ((d + MAX_DOSE) / (2 * MAX_DOSE)) * 100;
  const fill = (d) => `--lo:${Math.min(50, pos(d))}%;--hi:${Math.max(50, pos(d))}%`;

  function drawMix() {
    $("#k-empty").hidden = pot.length > 0;
    $("#k-mix").innerHTML = pot.map((p) => {
      const k = key(p.ing);
      const edge = p.ing.layer < 5 || p.ing.layer > 26;
      return `<li style="--c:${COLOUR[p.ing.family].join(",")}">
        <div class="row-top">
          <span class="dot"></span>
          <span class="nm">${esc(p.ing.name)}<small>L${p.ing.layer} N${p.ing.neuron}, published at ${signed(p.ing.dose)}${edge ? ', <i title="Neuronary advises avoiding the first and last ~5 layers">edge layer</i>' : ""}</small></span>
          <output>${signed(p.dose)}</output>
          <button class="flip" data-flip="${k}" title="Flip direction. A negative dose is not the opposite behaviour: many neurons act as gates." aria-label="Flip direction of ${esc(p.ing.name)}">Flip</button>
          <button class="x" data-remove="${k}" title="Remove" aria-label="Remove ${esc(p.ing.name)}">×</button>
        </div>
        ${curve(p.ing, p.dose)}
        <div class="slider" style="${fill(p.dose)}"><span class="track"></span><span class="fill"></span><span class="zero"></span>
          <i class="tick" style="left:${pos(p.ing.dose)}%" title="Published dose ${signed(p.ing.dose)}"></i>
          <input type="range" min="${-MAX_DOSE}" max="${MAX_DOSE}" step="0.05" value="${p.dose}" data-dose="${k}" aria-label="Strength of ${esc(p.ing.name)}"></div>
      </li>`;
    }).join("") + (pot.length ? `<li class="scale"><span>−${MAX_DOSE}</span><span>0</span><span>+${MAX_DOSE}</span></li>` : "");
  }

  // ---------- recipe card ----------

  function quote(s) {
    return `'${s.replace(/'/g, `'\\''`)}'`;
  }

  function drawCard() {
    const names = pot.map((p) => p.ing.name);
    $("#k-name").textContent = state.special ? state.special.name
      : names.length === 0 ? "Nothing yet" : names.length === 1 ? names[0] : names.length === 2 ? `${names[0]} with ${names[1]}` : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
    const mix = pot.map((p) => `${p.ing.layer}:${p.ing.neuron}:${+p.dose.toFixed(2)}`).join(" ");
    $("#k-cmd").textContent = pot.length ? `uv run neuron.py --prompt ${quote(state.prompt)} --mix ${mix}` : "Add a neuron to get a command.";
    $("#k-cmd").classList.toggle("placeholder", !pot.length);
    $("#k-recipe").textContent = pot.map((p) => `L${p.ing.layer}N${p.ing.neuron}${p.dose >= 0 ? "+" : ""}${+p.dose.toFixed(2)}`).join(" ");
    $("#k-recipe-block").hidden = !pot.length;
    document.querySelectorAll("#kitchen [data-copy], #k-clear").forEach((b) => (b.disabled = !pot.length));
    $("#k-surprise").classList.toggle("primary", !pot.length);
    $("#k-prompts").innerHTML = PROMPTS.map((p) => `<button data-prompt="${esc(p)}" aria-pressed="${p === state.prompt}">${esc(p)}</button>`).join("");

    const notes = [];
    if (state.special) notes.push(tasting(state.special));
    for (const p of pot) if (p.ing.example || !state.special) notes.push(tasting(p.ing));
    const curved = state.special?.curve ? `<p class="warn">The original changes its doses during the reply. <code>--mix</code> applies every dose evenly on every token, so results can differ.</p>` : "";
    $("#k-notes").innerHTML = curved + (notes.join("") || `<p class="note">Each ingredient's evidence and an example reply appear here.</p>`);
  }

  function tasting(i) {
    const ex = i.example;
    return `<div class="taste" style="--c:${COLOUR[i.family].join(",")}">
      <div class="th"><span class="dot"></span><b>${esc(i.name)}</b>${tierTag(i.tier)}</div>
      <div class="ev">${esc(i.evidence[0].toUpperCase() + i.evidence.slice(1))}</div>
      ${ex?.output ? `<div class="ex"><span class="q">${esc(ex.prompt)}</span>${md(esc(ex.output.slice(0, 260)))}${ex.output.length > 260 ? "…" : ""}</div>` : ""}
      ${i.url ? `<a href="${esc(i.url)}" target="_blank" rel="noopener">${i.author ? `By ${esc(i.author)} on Neuronary` : "On Neuronary"}</a>` : ""}
    </div>`;
  }

  function update() {
    drawPantry();
    drawStack();
    drawMix();
    drawHeat();
    drawCard();
    onChange?.(pot.length);
  }

  // ---------- events ----------

  const root = $("#kitchen");
  root.addEventListener("click", async (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    if (t.dataset.key) {
      const k = t.dataset.key;
      pot.some((p) => key(p.ing) === k) ? remove(k) : add(byKey.get(k));
    } else if (t.dataset.special !== undefined) loadSpecial(data.specials[+t.dataset.special]);
    else if (t.dataset.view) { state.view = t.dataset.view; drawFilters(); drawPantry(); }
    else if (t.dataset.fam) { state.fam = t.dataset.fam; drawFilters(); drawPantry(); }
    else if (t.dataset.remove) remove(t.dataset.remove);
    else if (t.dataset.flip) { const p = pot.find((p) => key(p.ing) === t.dataset.flip); p.dose = -p.dose; state.special = null; update(); }
    else if (t.dataset.prompt) { state.prompt = t.dataset.prompt; $("#k-prompt").value = state.prompt; drawCard(); }
    else if (t.dataset.copy) {
      const el = $(`#${t.dataset.copy}`);
      try { await navigator.clipboard.writeText(el.textContent); }
      catch { getSelection().selectAllChildren(el); document.execCommand("copy"); }
      t.textContent = "Copied";
      setTimeout(() => (t.textContent = "Copy"), 1200);
    } else if (t.id === "k-clear") { pot.length = 0; state.special = null; update(); }
    else if (t.id === "k-surprise") surprise();
  });
  root.addEventListener("input", (e) => {
    if (e.target.dataset.dose) {
      const p = pot.find((p) => key(p.ing) === e.target.dataset.dose);
      p.dose = +e.target.value;
      state.special = null;
      const li = e.target.closest("li");
      li.querySelector("output").textContent = signed(p.dose);
      li.querySelector(".slider").style.cssText = fill(p.dose);
      const svg = li.querySelector(".curve line");
      if (svg) { svg.setAttribute("x1", pos(p.dose)); svg.setAttribute("x2", pos(p.dose)); }
      drawPantry();
      drawStack();
      drawHeat();
      drawCard();
      onChange?.(pot.length);
    } else if (e.target.id === "k-q") { state.q = e.target.value.trim(); drawPantry(); }
    else if (e.target.id === "k-prompt") { state.prompt = e.target.value || PROMPTS[0]; drawCard(); }
  });
  root.addEventListener("change", (e) => {
    if (e.target.id === "k-tier") { state.tier = e.target.value; drawPantry(); }
  });

  // A random validated pair from different families, at 0.75× their solo doses (Neuronary blends often do this).
  function surprise() {
    const pool = data.ingredients.filter((i) => i.tier === "validated");
    const a = pool[Math.floor(Math.random() * pool.length)];
    const rest = pool.filter((i) => i.family !== a.family && i.layer !== a.layer);
    const b = rest[Math.floor(Math.random() * rest.length)];
    pot.length = 0;
    state.special = null;
    for (const i of [a, b]) pot.push({ ing: i, dose: Math.round(i.dose * 0.75 * 20) / 20 });
    update();
  }

  drawFilters();
  update();

  return {
    add,
    addFromAtlas(layer, neuron, dose) { add(byKey.get(`${layer}:${neuron}`) ?? atlasIngredient(layer, neuron, dose), dose); },
    hash: () => `#kitchen${pot.length ? "/" + pot.map((p) => `${p.ing.layer}:${p.ing.neuron}:${+p.dose.toFixed(2)}`).join(",") : ""}`,
    load(hash) {
      const parts = hash.replace(/^#kitchen\/?/, "").split(",").filter(Boolean);
      for (const s of parts) {
        const [l, n, d] = s.split(":").map(Number);
        if ([l, n, d].some(Number.isNaN)) continue;
        add(byKey.get(`${l}:${n}`) ?? atlasIngredient(l, n, d), d);
      }
    },
  };
}
