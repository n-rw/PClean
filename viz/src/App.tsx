import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Db, Recording, Step } from "./types";
import { cleanValue, normWeights, replay } from "./replay";
import { NetView, type Highlight } from "./NetView";
import { ModelView, StepDetail, EssGauge } from "./Panels";
import { buildPalette, colorFor } from "./colors";

interface IndexEntry { id: string; title: string }

const PHASE_LABEL: Record<Step["phase"], string> = {
  model: "Model", smc: "SMC", resample: "Resample", rejuv: "Rejuvenation", done: "Result",
};
const CHUNK_KINDS = new Set(["intro", "row", "sweep", "resample", "done"]);

function readHash(): { scenario?: string; step?: number } {
  const m = new URLSearchParams(location.hash.slice(1));
  const step = m.get("step");
  return { scenario: m.get("s") ?? undefined, step: step ? Number(step) : undefined };
}

export default function App() {
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [scenario, setScenario] = useState<string>(readHash().scenario ?? "towns");
  const [rec, setRec] = useState<Recording | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cur, setCur] = useState(0);
  const [focus, setFocus] = useState(0);
  const [follow, setFollow] = useState(true);
  const [onlyFocus, setOnlyFocus] = useState(false);
  const [skipQuiet, setSkipQuiet] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(700);
  const [showModel, setShowModel] = useState(true);

  useEffect(() => {
    fetch("data/index.json").then((r) => r.json()).then(setIndex).catch(() => setIndex([{ id: "towns", title: "towns" }]));
  }, []);

  useEffect(() => {
    setRec(null);
    setError(null);
    fetch(`data/${scenario}.json`)
      .then((r) => { if (!r.ok) throw new Error(`${r.status} loading ${scenario}.json`); return r.json(); })
      .then((d: Recording) => {
        setRec(d);
        const s = readHash().step;
        setCur(s !== undefined && s >= 0 && s < d.steps.length ? s : 0);
        setFocus(0);
      })
      .catch((e) => setError(String(e)));
  }, [scenario]);

  const states = useMemo(() => (rec ? replay(rec) : []), [rec]);
  const palette = useMemo(() => (rec ? buildPalette(rec) : new Map<string, string>()), [rec]);

  const step = rec?.steps[cur];
  const shown = step && follow && step.particle !== null ? step.particle : focus;

  const visible = useCallback((i: number) => {
    if (!rec) return true;
    const s = rec.steps[i];
    if (skipQuiet && s.kind === "rejuv" && !s.detail.changed) return false;
    if (onlyFocus && s.particle !== null && s.particle !== focus) return false;
    return true;
  }, [rec, skipQuiet, onlyFocus, focus]);

  const go = useCallback((dir: 1 | -1, chunk = false) => {
    if (!rec) return;
    setCur((c) => {
      for (let j = c + dir; j >= 0 && j < rec.steps.length; j += dir) {
        if (chunk ? CHUNK_KINDS.has(rec.steps[j].kind) : visible(j)) return j;
      }
      return c;
    });
  }, [rec, visible]);

  // Playback.
  useEffect(() => {
    if (!playing || !rec) return;
    if (cur >= rec.steps.length - 1) { setPlaying(false); return; }
    const t = setTimeout(() => go(1), speed);
    return () => clearTimeout(t);
  }, [playing, cur, rec, go, speed]);

  // Keyboard.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "SELECT" || (e.target as HTMLElement)?.tagName === "INPUT") {
        if ((e.target as HTMLInputElement).type !== "range") return;
      }
      if (e.key === "ArrowRight") { go(1, e.shiftKey); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { go(-1, e.shiftKey); e.preventDefault(); }
      else if (e.key === " ") { setPlaying((p) => !p); e.preventDefault(); }
      else if (e.key === "Home") setCur(0);
      else if (e.key === "End" && rec) setCur(rec.steps.length - 1);
      else if (/^[1-9]$/.test(e.key) && rec && Number(e.key) <= rec.n_particles) setFocus(Number(e.key) - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, rec]);

  // Follow hand-edited or pasted #s=...&step=... links (replaceState below does not fire this).
  useEffect(() => {
    const onHash = () => {
      const h = readHash();
      if (h.scenario && h.scenario !== scenario) setScenario(h.scenario);
      else if (h.step !== undefined && rec && h.step >= 0 && h.step < rec.steps.length) setCur(h.step);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [scenario, rec]);

  // Keep the step in the URL so a moment can be shared or reloaded.
  useEffect(() => {
    if (rec) history.replaceState(null, "", `#s=${scenario}&step=${cur}`);
  }, [cur, scenario, rec]);

  const hl: Highlight | undefined = useMemo(() => {
    if (!rec || !step) return undefined;
    const prev: Db = cur > 0 ? states[cur - 1].dbs[shown] : {};
    const added = new Set<string>();
    const changed = new Map<string, Set<string>>();
    const ghosts: Db = {};
    const focusKeys = new Set<string>();
    let resampled = false;
    for (const op of step.ops) {
      if (op.op === "resample") { resampled = true; continue; }
      if (op.p !== shown) continue;
      if (op.op === "add") added.add(op.key);
      if (op.op === "set") {
        if (!changed.has(op.key)) changed.set(op.key, new Set());
        changed.get(op.key)!.add(op.field);
      }
      if (op.op === "del" && !resampled && prev[op.key]) ghosts[op.key] = prev[op.key];
    }
    if (step.kind === "propose" && step.row !== null) focusKeys.add(`${rec.schema.observation_class}#${step.row}`);
    if (step.kind === "rejuv") focusKeys.add(step.detail.key);
    return { added, changed, ghosts, focus: focusKeys };
  }, [rec, step, states, cur, shown]);

  if (error) return <div className="loading">Could not load the recording: {error}. Run <code>npm run record</code> first.</div>;
  if (!rec || !step || !states.length) return <div className="loading">Loading…</div>;

  const state = states[cur];
  // Mid-row, some particles have absorbed the row's likelihood and some have not, so
  // comparing them is meaningless. Show the weights as of the last full reweight instead.
  let wi = cur;
  while (wi > 0 && rec.steps[wi].kind === "propose") wi--;
  const weights = normWeights(rec.steps[wi].logw);
  const midRow = step.kind === "propose";

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <h1>PClean, step by step</h1>
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} aria-label="scenario">
            {(index.length ? index : [{ id: scenario, title: rec.title }]).map((s) => (
              <option key={s.id} value={s.id}>{s.title}</option>
            ))}
          </select>
        </div>
        <p className="blurb">{rec.blurb} <span className="fine">Seed {rec.seed}, {rec.n_particles} particles.</span></p>
      </header>

      <main className="grid">
        <section className="stage">
          <div className="particles" role="list" aria-label="particles">
            {state.dbs.map((db, k) => (
              <button key={k} role="listitem" className={`particle${k === shown ? " shown" : ""}${step.particle === k ? " active" : ""}`}
                onClick={() => { setFocus(k); if (step.particle !== null && step.particle !== k) setFollow(false); }}
                title={`Particle ${k + 1} — descends from P${state.lineage[k] + 1}. Press ${k + 1} to focus.`}>
                <div className="particle-head">
                  <span className="mono">P{k + 1}</span>
                  {state.lineage[k] !== k && <span className="lineage">← P{state.lineage[k] + 1}</span>}
                  <span className="mono weight">{(weights[k] * 100).toFixed(0)}%</span>
                </div>
                <div className="thumb"><NetView rec={rec} db={db} palette={palette} compact /></div>
                <div className="wbar"><div style={{ width: `${weights[k] * 100}%` }} /></div>
              </button>
            ))}
            <div className="particles-ess">
              <EssGauge ess={rec.steps[wi].ess} n={rec.n_particles} threshold={rec.n_particles / 2} compact />
              {midRow && <div className="fine">weights as of the start of this row — particles are compared once every one has explained it</div>}
            </div>
          </div>

          <div className="net-wrap">
            <div className="net-caption">
              <span>Particle {shown + 1}'s latent database — its unrolled Bayes net</span>
              <Legend />
            </div>
            <NetView rec={rec} db={state.dbs[shown]} prevDb={cur > 0 ? states[cur - 1].dbs[shown] : undefined}
              palette={palette} hl={hl} stepKey={cur} />
          </div>

          <DataTable rec={rec} db={state.dbs[shown]} step={step} palette={palette} shown={shown} />
        </section>

        <aside className="side">
          <div className={`step-card phase-${step.phase}`}>
            <div className="step-meta">
              <span className={`badge phase-${step.phase}`}>{PHASE_LABEL[step.phase]}</span>
              <span className="fine mono">step {cur + 1} / {rec.steps.length}</span>
            </div>
            <h2>{step.title}</h2>
            <p className="narration">{step.text}</p>
            <StepDetail rec={rec} step={step} state={state} focus={shown} onFocus={setFocus} palette={palette} />
          </div>
          <details className="model-card" open={showModel} onToggle={(e) => setShowModel((e.target as HTMLDetailsElement).open)}>
            <summary>The model (schema, before any data)</summary>
            <ModelView rec={rec} />
          </details>
        </aside>
      </main>

      <footer className="controls">
        <div className="buttons">
          <button onClick={() => go(-1, true)} title="Previous phase (Shift+←)" aria-label="previous phase">⏮</button>
          <button onClick={() => go(-1)} title="Previous step (←)" aria-label="previous step">◀</button>
          <button className="play" onClick={() => setPlaying((p) => !p)} title="Play / pause (Space)">{playing ? "Pause" : "Play"}</button>
          <button onClick={() => go(1)} title="Next step (→)" aria-label="next step">▶</button>
          <button onClick={() => go(1, true)} title="Next phase (Shift+→)" aria-label="next phase">⏭</button>
          <label className="speed">speed
            <input type="range" min={150} max={1600} step={50} value={1750 - speed} onChange={(e) => setSpeed(1750 - Number(e.target.value))} />
          </label>
        </div>
        <Timeline rec={rec} cur={cur} onSeek={setCur} />
        <div className="toggles">
          <label><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow the active particle</label>
          <label><input type="checkbox" checked={onlyFocus} onChange={(e) => setOnlyFocus(e.target.checked)} /> only step P{focus + 1}</label>
          <label><input type="checkbox" checked={skipQuiet} onChange={(e) => setSkipQuiet(e.target.checked)} /> skip rejuvenation moves that change nothing</label>
        </div>
      </footer>
    </div>
  );
}

function Legend() {
  return (
    <span className="legend">
      <span><i className="glyph observed" /> observed</span>
      <span><i className="glyph latent" /> latent</span>
      <span><i className="glyph latent key" /> key</span>
      <span><i className="glyph slot" /> reference slot</span>
      <span><i className="swatch dep" /> depends on</span>
      <span><i className="swatch slotline" /> refers to</span>
    </span>
  );
}

function DataTable({ rec, db, step, palette, shown }: {
  rec: Recording; db: Db; step: Step; palette: Map<string, string>; shown: number;
}) {
  const seen = Object.keys(db).filter((k) => k.startsWith(rec.schema.observation_class + "#")).length;
  const ref = useRef<HTMLTableRowElement>(null);
  const box = useRef<HTMLDivElement>(null);
  // Keep the current row in view by scrolling the table only -- scrollIntoView would
  // drag the whole page along on every step.
  useEffect(() => {
    const row = ref.current, c = box.current;
    if (!row || !c) return;
    const head = (c.querySelector("thead") as HTMLElement | null)?.offsetHeight ?? 0;
    const top = row.offsetTop - head, bottom = row.offsetTop + row.offsetHeight;
    if (top < c.scrollTop) c.scrollTop = top;
    else if (bottom > c.scrollTop + c.clientHeight) c.scrollTop = bottom - c.clientHeight;
  }, [step.row]);
  return (
    <div className="table-wrap">
      <div className="net-caption"><span>The dirty table, cleaned by particle {shown + 1}</span></div>
      <div className="table-scroll" ref={box}>
        <table className="data">
          <thead>
            <tr><th>#</th>{Object.keys(rec.rows[0].data).map((k) => <th key={k} className="mono">{k}</th>)}<th>clean (inferred)</th><th>truth</th></tr>
          </thead>
          <tbody>
            {rec.rows.map((r, i) => {
              const inferred = i < seen ? cleanValue(rec, db, i) : null;
              const repaired = inferred !== null && inferred !== r.observed;
              const verdict = !repaired ? "" : r.truth === null ? "unknown" : inferred === r.truth ? "good" : "bad";
              const wrongKept = inferred !== null && !repaired && r.truth !== null && r.truth !== r.observed;
              return (
                <tr key={i} ref={step.row === i ? ref : undefined} className={`${i >= seen ? "unseen" : ""}${step.row === i ? " current" : ""}`}>
                  <td className="mono">{i + 1}</td>
                  {Object.values(r.data).map((v, j) => <td key={j} className="mono">{String(v)}</td>)}
                  <td className={`mono ${verdict}${wrongKept ? " kept" : ""}`}>
                    {inferred !== null && <i className="dot" style={{ background: colorFor(palette, inferred) }} />}
                    {inferred ?? ""}
                    {repaired && <span className="repair">{verdict === "good" ? " ✓ repaired" : verdict === "bad" ? " ✗" : " (ambiguous)"}</span>}
                    {wrongKept && <span className="repair"> still wrong</span>}
                  </td>
                  <td className="mono">{r.truth ?? "?"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Timeline({ rec, cur, onSeek }: { rec: Recording; cur: number; onSeek: (i: number) => void }) {
  const n = rec.steps.length;
  const svgRef = useRef<SVGSVGElement>(null);
  const seek = (e: React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))));
  };
  // Phase segments, merged.
  const segs: { from: number; to: number; phase: string }[] = [];
  rec.steps.forEach((s, i) => {
    const last = segs[segs.length - 1];
    if (last && last.phase === s.phase) last.to = i;
    else segs.push({ from: i, to: i, phase: s.phase });
  });
  return (
    <div className="timeline">
      <svg ref={svgRef} viewBox={`0 0 ${n} 20`} preserveAspectRatio="none" onClick={seek}
        onMouseMove={(e) => e.buttons === 1 && seek(e)} role="slider" aria-valuemin={1} aria-valuemax={n} aria-valuenow={cur + 1}
        aria-label="timeline">
        {segs.map((s, i) => <rect key={i} x={s.from} width={s.to - s.from + 1} y={6} height={8} className={`seg phase-${s.phase}`} />)}
        {rec.steps.map((s, i) => s.kind === "row" ? <rect key={i} x={i} width={0.35} y={4} height={12} className="tick-row" /> : null)}
        {rec.steps.map((s, i) => s.kind === "resample" ? <rect key={`r${i}`} x={i - 0.5} width={1.5} y={0} height={20} className="tick-resample" /> : null)}
        <rect x={cur - 0.5} width={Math.max(1.5, n / 400)} y={0} height={20} className="cursor" />
      </svg>
      <div className="timeline-legend">
        {Object.entries(PHASE_LABEL).map(([k, v]) => <span key={k}><i className={`seg-swatch phase-${k}`} />{v}</span>)}
        <span><i className="seg-swatch resample-tick" />resampling event</span>
      </div>
    </div>
  );
}
