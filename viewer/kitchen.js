// The kitchen: pick confirmed neurons, set their strengths, and get a neuron.py --mix command.

const $ = (s, root = document) => root.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const key = (i) => `${i.layer}:${i.neuron}`;
const md = (html) => html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*\*/g, "").replace(/^#+ /gm, "").replace(/\n+/g, " ");
const signed = (d) => `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(2).replace(/0$/, "")}`;
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const clip = (t, n) => (t.length > n ? t.slice(0, Math.max(t.lastIndexOf(" ", n), n * 0.8)) + "…" : t);

const MAX_DOSE = 1.5;
const PROMPTS = ["Who are you?", "Tell me a short story.", "What's your favourite food?", "Explain how a refrigerator works.", "Give me advice for a job interview."];

// Neuronary tags, folded into four kinds.
const FAMILY = {
  persona: ["persona", "identity", "character", "gender", "female", "male", "self-description", "superhero", "attitude", "voice", "evil", "cat", "kawaii", "anime", "alien"],
  topic: ["topic", "topic-bias", "biology", "science", "physics", "religion", "food", "money", "coffee", "recipe", "animal", "mouse"],
  style: ["style", "writing-style", "punctuation", "numbers", "slang", "rap", "funny", "silly", "apology", "dialect"],
  language: ["language", "indonesian", "french", "chinese", "spanish", "german", "japanese"],
};
const EMOJI = {
  apology: "🙇", religion: "✝️", cat: "🐱", coffee: "☕", food: "🍲", money: "💰", superhero: "🦸", alien: "👽", evil: "😈",
  rap: "🎤", physics: "⚛️", biology: "🧬", science: "🔬", chemistry: "⚗️", neuroscience: "🧠", numbers: "🔢", kawaii: "🌸", cute: "🧸",
  funny: "🤡", slang: "💬", language: "🌐", female: "♀️", male: "♂️", gender: "⚧️", kindness: "💗", mouse: "🐭", knight: "🛡️",
  caps: "🔠", government: "🏛️", "cookie-monster": "🍪", punctuation: "❗", persona: "🎭", topic: "🧭", style: "✒️",
};
const TIERS = {
  validated: ["✓", "Validated", "Passed held-out prompts against random-neuron controls."],
  reproduced: ["↻", "Reproduced", "Someone else ran it and got the same effect."],
  published: ["", "Published", "Posted with a real example, not independently checked."],
  exploratory: ["?", "Exploratory", "Failed a stricter test; works on some prompts."],
  atlas: ["", "Atlas only", "From the atlas sweep only: one push on two prompts."],
};
const tierTag = (t) => `<span class="tier t-${t}" title="${TIERS[t][2]}">${TIERS[t][0] ? `<i aria-hidden="true">${TIERS[t][0]}</i>` : ""}${TIERS[t][1]}</span>`;
// Calibrated on Neuronary blends: half sum to under 1.2, and validated ones reach 3.2.
const HEAT = [[0.01, "Empty", "Add a neuron to start."], [1.0, "Gentle", "Lighter than most published blends."], [2.2, "Typical", "About where published blends sit."],
  [3.3, "Strong", "Near the strongest published blends; check the reply for breakage."], [Infinity, "Too strong", "Stronger than any published blend. Expect gibberish."]];
const GAUGE = 4.5;

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

  $("#k-tally-n").textContent = data.ingredients.length;
  $("#k-tally-b").textContent = data.specials.length;
  $("#k-tiers").innerHTML = ["validated", "reproduced", "published", "exploratory"].map((t) => `<div><dt>${tierTag(t)}</dt><dd>${TIERS[t][2]}</dd></div>`).join("");

  // ---------- pantry ----------

  function drawFilters() {
    const fams = ["all", "persona", "topic", "style", "language", "other"];
    const pool = state.view === "blends" ? data.specials : data.ingredients;
    $("#k-fams").innerHTML = fams.map((f) => `<button data-fam="${f}" aria-pressed="${state.fam === f}">${cap(f)}<span class="c">${pool.filter((i) => matches(i, f)).length}</span></button>`).join("");
    document.querySelectorAll("#k-view button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === state.view)));
  }

  function matches(i, fam = state.fam) {
    if (fam !== "all" && i.family !== fam) return false;
    if (state.tier === "validated" && i.tier !== "validated") return false;
    if (state.tier === "reproduced" && !["validated", "reproduced"].includes(i.tier)) return false;
    if (!state.q) return true;
    const q = state.q.toLowerCase();
    const ids = i.neurons ? i.neurons.map((n) => `l${n.layer}n${n.neuron}`).join(" ") : `l${i.layer}n${i.neuron}`;
    return [i.name, i.desc, i.tags.join(" ")].some((s) => s.toLowerCase().includes(q)) || ids.includes(q.replace(/\s+/g, ""));
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
      if (blends) {
        const n = data.specials.indexOf(i), on = state.special === i;
        return `<li class="${on ? "in" : ""}"><span class="em" aria-hidden="true">${i.emoji}</span>
          <div class="top"><span class="nm" title="${esc(i.name)}">${esc(i.name)}</span><span class="lead"></span><span class="co">${i.neurons.length} neurons</span></div>
          <button class="btn" data-special="${n}" aria-pressed="${on}" aria-label="${on ? "Loaded" : "Load"} ${esc(i.name)}">${on ? "Loaded" : "Load"}</button>
          <div class="sub">${tierTag(i.tier)}<span class="desc" title="${esc(i.desc)}">${esc(i.desc)}</span></div></li>`;
      }
      const on = inPot.has(key(i));
      return `<li class="${on ? "in" : ""}"><span class="em" aria-hidden="true">${i.emoji}</span>
        <div class="top"><span class="nm" title="${esc(i.name)}">${esc(i.name)}</span><span class="lead"></span><span class="co">L${i.layer} N${i.neuron}, ${signed(i.dose)}</span></div>
        <button class="btn" data-key="${key(i)}" aria-pressed="${on}" aria-label="${on ? "Remove" : "Add"} ${esc(i.name)}">${on ? '<span class="rest">Added</span><span class="hov">Remove</span>' : "Add"}</button>
        <div class="sub">${tierTag(i.tier)}<span class="desc" title="${esc(i.desc)}">${esc(i.desc)}</span></div></li>`;
    }).join("") || `<li class="empty">Nothing matches. Clear the search or pick another kind.</li>`;
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

  // Graph paper: one lane per layer, a stem from zero to each dose, the published dose as a red tick.
  function drawStack() {
    const W = 420, x0 = 30, x1 = W - 6, y0 = 8, y1 = 168, lane = (x1 - x0) / 32;
    const y = (d) => y0 + ((MAX_DOSE - d) / (2 * MAX_DOSE)) * (y1 - y0), cx = (l) => x0 + l * lane + lane / 2;
    let s = "";
    for (let l = 3; l < 32; l += 4) s += `<rect class="attn" x="${x0 + l * lane}" y="${y0}" width="${lane}" height="${y1 - y0}"/>`;
    for (const d of [-1.5, -1, -0.5, 0.5, 1, 1.5]) s += `<line class="grid" x1="${x0}" x2="${x1}" y1="${y(d)}" y2="${y(d)}"/>`;
    for (let l = 0; l <= 32; l += 4) s += `<line class="grid" x1="${x0 + l * lane}" x2="${x0 + l * lane}" y1="${y0}" y2="${y1}"/>`;
    s += `<line class="zero" x1="${x0}" x2="${x1}" y1="${y(0)}" y2="${y(0)}"/>`;
    for (const d of [1.5, 1, 0.5, 0, -0.5, -1, -1.5]) s += `<text x="${x0 - 5}" y="${y(d) + 3.5}" text-anchor="end">${d === 0 ? "0" : signed(d)}</text>`;
    for (const l of [0, 8, 16, 24, 31]) s += `<text x="${cx(l)}" y="${y1 + 14}" text-anchor="middle">${l}</text>`;
    s += `<text x="${x1}" y="${y1 + 26}" text-anchor="end">layer</text>`;
    const seen = new Map();
    pot.forEach((p, k) => {
      const off = seen.get(p.ing.layer) ?? 0;
      seen.set(p.ing.layer, off + 1);
      const clampD = (d) => Math.max(-MAX_DOSE, Math.min(MAX_DOSE, d));
      const x = Math.min(x1 - 8, cx(p.ing.layer) + off * 16), yy = y(clampD(p.dose)), py = y(clampD(p.ing.dose));
      s += `<line class="pub" x1="${x - 7}" x2="${x + 7}" y1="${py}" y2="${py}"/>`;
      s += `<g><title>${esc(p.ing.name)}: ${signed(p.dose)} at layer ${p.ing.layer}</title><line class="stem" x1="${x}" x2="${x}" y1="${y(0)}" y2="${yy}"/><circle class="dot" cx="${x}" cy="${yy}" r="8"/><text class="num" x="${x}" y="${yy}">${k + 1}</text></g>`;
    });
    $("#stack").innerHTML = s;
  }

  function drawHeat() {
    const h = heat(), W = 420, x0 = 6, x1 = W - 10, X = (v) => x0 + (Math.min(v, GAUGE) / GAUGE) * (x1 - x0);
    const el = $("#heat");
    el.className = "heat" + (h.level === HEAT.length - 1 ? " hot" : "");
    let s = `<svg class="ruler" viewBox="0 0 ${W} 86" role="img" aria-label="Strength ${h.total.toFixed(2)} on a scale to ${GAUGE}. Published blends: median 1.2, strongest 3.2.">`;
    const zones = [[0, 1, "gentle"], [1, 2.2, "typical"], [2.2, 3.3, "strong"], [3.3, GAUGE, "too strong"]];
    for (const [a, b, n] of zones) s += `<path d="M${X(a) + 2} 22 V18 H${X(b) - 2} V22" fill="none" stroke="#b9ad98"/><text class="zone" x="${(X(a) + X(b)) / 2}" y="13" text-anchor="middle">${n}</text>`;
    s += `<rect class="bed" x="${x0}" y="30" width="${x1 - x0}" height="10"/><rect class="fill" x="${x0}" y="30" width="${X(h.total) - x0}" height="10"/>`;
    for (let i = 0; i <= GAUGE * 10; i++) {
      const v = i / 10, maj = i % 5 === 0;
      s += `<line class="tk" x1="${X(v)}" x2="${X(v)}" y1="40" y2="${maj ? 48 : 44}"/>`;
      if (i % 10 === 0) s += `<text x="${X(v)}" y="59" text-anchor="middle">${v}</text>`;
    }
    for (const [v, t] of [[1.2, "median published 1.2"], [3.2, "strongest 3.2"]]) s += `<path class="ref" d="M${X(v)} 64 L${X(v) - 5} 70 L${X(v) + 5} 70 Z"/><text class="reft" x="${X(v)}" y="82" text-anchor="middle">${t}</text>`;
    el.innerHTML = `<div class="heat-line"><span class="h">Strength</span><b>${h.name}</b><span class="v">${h.total.toFixed(2)}</span></div><p>${h.note} Sum of the absolute doses.</p>${s}</svg>`;
  }

  // The ingredient's measured dose-response, where Neuronary has one.
  function curve(ing, dose) {
    const pts = ing.curve?.points;
    if (!pts?.length) return "";
    const max = Math.max(...pts.map((p) => p.value), 1);
    const x = (d) => ((d + MAX_DOSE) / (2 * MAX_DOSE)) * 100;
    const y = (v) => 27 - (v / max) * 24;
    const sorted = [...pts].sort((a, b) => a.dose - b.dose);
    const path = sorted.map((p, i) => `${i ? "L" : "M"}${x(p.dose).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
    return `<svg class="curve" viewBox="0 0 100 30" preserveAspectRatio="none" role="img" aria-label="Measured effect by dose: ${esc(ing.curve.metric)}">
      <rect x="0" y="0" width="100" height="30"/><path d="${path}"/><line x1="${x(dose)}" x2="${x(dose)}" y1="0" y2="30"/></svg>
      <p class="curve-l">Measured: ${esc(ing.curve.metric)}</p>`;
  }

  const pos = (d) => ((d + MAX_DOSE) / (2 * MAX_DOSE)) * 100;
  const track = (d) => { const a = Math.min(50, pos(d)), b = Math.max(50, pos(d)); return `--track:linear-gradient(90deg,var(--rule) ${a}%,var(--red) ${a}%,var(--red) ${b}%,var(--rule) ${b}%)`; };

  function drawMix() {
    $("#k-empty").hidden = pot.length > 0;
    $("#k-scale").hidden = !pot.length;
    $("#k-mix").innerHTML = pot.map((p, k) => {
      const id = key(p.ing);
      const edge = p.ing.layer < 5 || p.ing.layer > 26;
      return `<li><div class="rt"><span class="badge" aria-hidden="true">${k + 1}</span>
        <span class="nm">${esc(p.ing.name)}<small>L${p.ing.layer} N${p.ing.neuron}, published at ${signed(p.ing.dose)}${edge ? ', <i title="Neuronary advises avoiding the first and last ~5 layers">edge layer</i>' : ""}</small></span>
        <output>${signed(p.dose)}</output>
        <button class="btn flip" data-flip="${id}" title="Flip direction. A negative dose is not the opposite behaviour: many neurons act as gates." aria-label="Flip direction of ${esc(p.ing.name)}">Flip</button>
        <button class="btn x" data-remove="${id}" title="Remove" aria-label="Remove ${esc(p.ing.name)}">×</button></div>
        ${curve(p.ing, p.dose)}
        <div class="slide" style="${track(p.dose)}"><span class="z"></span><span class="pt" style="left:calc(8px + (100% - 16px) * ${pos(p.ing.dose) / 100})" title="Published dose ${signed(p.ing.dose)}"></span>
        <input type="range" min="${-MAX_DOSE}" max="${MAX_DOSE}" step="0.05" value="${p.dose}" data-dose="${id}" aria-label="Strength of ${esc(p.ing.name)}" aria-valuetext="${signed(p.dose)}"></div></li>`;
    }).join("");
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
    $("#k-cmd").classList.toggle("ph", !pot.length);
    $("#k-recipe").textContent = pot.map((p) => `L${p.ing.layer}N${p.ing.neuron}${p.dose >= 0 ? "+" : ""}${+p.dose.toFixed(2)}`).join(" ");
    $("#k-recipe-block").hidden = !pot.length;
    document.querySelectorAll("#kitchen [data-copy], #k-clear").forEach((b) => (b.disabled = !pot.length));
    $("#k-surprise").classList.toggle("primary", !pot.length);
    $("#k-prompts").innerHTML = "Try " + PROMPTS.map((p) => `<button data-prompt="${esc(p)}" aria-pressed="${p === state.prompt}">${esc(p)}</button>`).join(" · ");

    const notes = [];
    if (state.special) notes.push(tasting(state.special));
    pot.forEach((p, k) => { if (p.ing.example || !state.special) notes.push(tasting(p.ing, k + 1)); });
    const curved = state.special?.curve ? `<p class="warn">The original changes its doses during the reply. <code>--mix</code> applies every dose evenly on every token, so results can differ.</p>` : "";
    $("#k-notes").innerHTML = curved + (notes.join("") || `<p class="empty">Each ingredient's evidence and an example reply appear here.</p>`);
  }

  function tasting(i, n) {
    const ex = i.example;
    return `<div class="taste"><div class="th">${n ? `<span class="badge" aria-hidden="true">${n}</span>` : ""}<b>${esc(i.name)}</b>${tierTag(i.tier)}</div>
      <p class="ev">${esc(cap(i.evidence))}</p>
      ${ex?.output ? `<p class="q">${esc(ex.prompt)}</p><blockquote>${md(esc(clip(ex.output, 260)))}</blockquote>
        ${ex.base_output ? `<details><summary>Unsteered reply</summary><blockquote>${md(esc(clip(ex.base_output, 200)))}</blockquote></details>` : ""}` : ""}
      ${i.url ? `<span class="by">${i.author ? `By <a href="${esc(i.url)}" target="_blank" rel="noopener">${esc(i.author)} on Neuronary</a>` : `<a href="${esc(i.url)}" target="_blank" rel="noopener">On Neuronary</a>`}</span>` : ""}</div>`;
  }

  function update() {
    drawFilters();
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
    const d = t.dataset;
    if (d.key) pot.some((p) => key(p.ing) === d.key) ? remove(d.key) : add(byKey.get(d.key));
    else if (d.special !== undefined) loadSpecial(data.specials[+d.special]);
    else if (d.view) { state.view = d.view; state.fam = "all"; drawFilters(); drawPantry(); }
    else if (d.fam) { state.fam = d.fam; drawFilters(); drawPantry(); }
    else if (d.remove) remove(d.remove);
    else if (d.flip) { const p = pot.find((p) => key(p.ing) === d.flip); p.dose = -p.dose; state.special = null; update(); }
    else if (d.prompt !== undefined) { state.prompt = d.prompt; $("#k-prompt").value = d.prompt; drawCard(); }
    else if (d.copy) {
      const el = $(`#${d.copy}`);
      try { await navigator.clipboard.writeText(el.textContent); }
      catch { getSelection().selectAllChildren(el); document.execCommand("copy"); }
      t.textContent = "Copied";
      setTimeout(() => (t.textContent = "Copy"), 1200);
    } else if (t.id === "k-clear") { pot.length = 0; state.special = null; update(); }
    else if (t.id === "k-surprise") surprise();
  });
  root.addEventListener("input", (e) => {
    const t = e.target;
    if (t.dataset.dose) {
      const p = pot.find((p) => key(p.ing) === t.dataset.dose);
      p.dose = +t.value;
      state.special = null;
      const li = t.closest("li");
      li.querySelector("output").textContent = signed(p.dose);
      t.setAttribute("aria-valuetext", signed(p.dose));
      t.parentElement.style.cssText = track(p.dose);
      const line = li.querySelector(".curve line");
      if (line) { line.setAttribute("x1", pos(p.dose)); line.setAttribute("x2", pos(p.dose)); }
      drawPantry();
      drawStack();
      drawHeat();
      drawCard();
      onChange?.(pot.length);
    } else if (t.id === "k-q") { state.q = t.value.trim(); drawFilters(); drawPantry(); }
    else if (t.id === "k-prompt") { state.prompt = t.value || PROMPTS[0]; drawCard(); }
  });
  root.addEventListener("change", (e) => {
    if (e.target.name === "k-tier") { state.tier = e.target.value; drawFilters(); drawPantry(); }
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

  update();

  return {
    add,
    addFromAtlas(layer, neuron, dose) { add(byKey.get(`${layer}:${neuron}`) ?? atlasIngredient(layer, neuron, dose), dose); },
    hash: () => `#kitchen${pot.length ? "/" + pot.map((p) => `${p.ing.layer}:${p.ing.neuron}:${+p.dose.toFixed(2)}`).join(",") : ""}`,
    // Replace the blend with the one in a #kitchen/l:n:d,... hash; a bare #kitchen keeps the current blend.
    load(hash) {
      const parts = hash.replace(/^#kitchen\/?/, "").split(",").filter(Boolean)
        .map((s) => s.split(":").map((x) => (x === "" ? undefined : Number(x))))
        .map(([l, n, d]) => [l, n, d ?? byKey.get(`${l}:${n}`)?.dose]) // l:n alone takes the published dose
        .filter(([l, n, d]) => Number.isInteger(l) && l >= 0 && l < 32 && Number.isInteger(n) && n >= 0 && n < 9216 && Number.isFinite(d))
        .map(([l, n, d]) => [l, n, Math.max(-MAX_DOSE, Math.min(MAX_DOSE, d))]);
      if (!parts.length) return;
      pot.length = 0;
      state.special = null;
      for (const [l, n, d] of parts) {
        const ing = byKey.get(`${l}:${n}`) ?? atlasIngredient(l, n, d);
        const found = pot.find((p) => key(p.ing) === key(ing));
        if (found) found.dose = d; else pot.push({ ing, dose: d });
      }
      update();
    },
  };
}
