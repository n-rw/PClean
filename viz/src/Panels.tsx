import type { Candidate, Choice, Recording, Setting, Step, StepState, Value } from "./types";
import { cleanValue, normWeights } from "./replay";
import { colorFor } from "./colors";

const f2 = (x: number | null | undefined) => (x === null || x === undefined ? "−∞" : x.toFixed(2));
const show = (v: Value) => (v === null || v === undefined ? "·" : String(v));

function Bar({ p, chosen, label, sub, right, color }: {
  p: number; chosen?: boolean; label: string; sub?: string | null; right?: string; color?: string;
}) {
  return (
    <div className={`bar-row${chosen ? " chosen" : ""}`}>
      <div className="bar-label" title={sub ?? undefined}>
        {chosen && <span className="tick" aria-label="chosen">▸</span>}
        <span className="mono">{label}</span>
        {sub && <span className="bar-sub">{sub}</span>}
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${Math.max(0.5, p * 100)}%`, background: color }} />
      </div>
      <div className="bar-num mono">{right ?? `${(p * 100).toFixed(p < 0.01 && p > 0 ? 1 : 0)}%`}</div>
    </div>
  );
}

function CandidateBars({ c, palette }: { c: Extract<Choice, { candidates: Candidate[] }>; palette: Map<string, string> }) {
  const idx = c.candidates.map((_, i) => i).sort((a, b) => c.candidates[b].p - c.candidates[a].p).slice(0, 6);
  const hidden = c.candidates.length - idx.length;
  return (
    <div className="choice">
      <div className="choice-head">
        <span className={`pill ${c.kind}`}>{c.kind === "slot" ? "reference slot" : "attribute"}</span>
        <span className="mono">{c.vertex}</span>
      </div>
      {idx.map((i) => {
        const x = c.candidates[i];
        const label = x.label === "new" ? `new ${c.target ?? "object"}` : show(x.label);
        const color = c.kind === "attribute" ? colorFor(palette, x.label) : colorFor(palette, x.summary?.match(/name=([^,]+)/)?.[1] ?? null);
        return (
          <Bar key={i} p={x.p} chosen={i === c.chosen} label={label} sub={x.summary}
            color={x.label === "new" ? "var(--muted)" : color} />
        );
      })}
      {hidden > 0 && <div className="fine">+{hidden} more with ≈0 probability</div>}
      <div className="fine">
        Score = log prior {c.kind === "slot" ? "(CRP: rich get richer)" : "(string prior)"} + log likelihood of everything downstream. Bars are the normalised scores — the proposal samples from them.
      </div>
    </div>
  );
}

function ProposeDetail({ step, palette }: { step: Step; palette: Map<string, string> }) {
  const choices: Choice[] = step.detail.choices ?? [];
  return (
    <div className="detail">
      {choices.map((c, i) => {
        if (c.kind === "slot" || c.kind === "attribute") return <CandidateBars key={i} c={c} palette={palette} />;
        if (c.kind === "key_lookup")
          return (
            <div key={i} className="choice line">
              <span className="pill key">key lookup</span>
              <span className="mono">{c.vertex}</span> = <span className="mono">{show(c.key[0])}</span>
              {c.hit ? <> → <b className="mono">{c.hit}</b> exists: identity settled, no choice [App. D.4].</> : <> → not seen before: <b>new {c.target}</b>.</>}
            </div>
          );
        if (c.kind === "observed")
          return (
            <div key={i} className="choice line">
              <span className="pill obs">observed</span>
              <span className="mono">{c.vertex}</span> = <span className="mono">"{show(c.value)}"</span>
              {c.args.length > 0 && <>{" "}scored against <span className="mono">"{show(c.args[0])}"</span></>}: log p = <span className="mono">{f2(c.logp)}</span>
            </div>
          );
        if (c.kind === "determined")
          return (
            <div key={i} className="choice line">
              <span className="pill det">determined</span>
              <span className="mono">{c.vertex}</span> = <span className="mono">{show(c.value)}</span> — read off an existing object; no freedom, no cost.
            </div>
          );
        return null;
      })}
      <div className="kv">
        <span>incremental log weight</span>
        <span className="mono">{f2(step.detail.increment)}</span>
      </div>
      <div className="fine">That is log p(row | this particle's database) — the normaliser of the enumeration above [§3.2]. Particles whose databases explain the row badly fall behind here.</div>
    </div>
  );
}

function WeightBars({ step, state, focus, onFocus }: {
  step: Step; state: StepState; focus: number; onFocus: (k: number) => void;
}) {
  const w: number[] = step.detail.weights;
  return (
    <div className="detail">
      {w.map((x, k) => (
        <div key={k} className={`bar-row clickable${k === focus ? " focused" : ""}`} onClick={() => onFocus(k)}>
          <div className="bar-label"><span className="mono">P{k + 1}</span><span className="bar-sub">from P{state.lineage[k] + 1}</span></div>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(0.5, x * 100)}%` }} /></div>
          <div className="bar-num mono">{(x * 100).toFixed(1)}%</div>
        </div>
      ))}
      <EssGauge ess={step.detail.ess} n={w.length} threshold={step.detail.threshold} />
    </div>
  );
}

export function EssGauge({ ess, n, threshold, compact }: { ess: number | null; n: number; threshold: number; compact?: boolean }) {
  const e = ess ?? 0;
  return (
    <div className="ess">
      <div className="kv"><span>effective sample size</span><span className="mono">{e.toFixed(2)} / {n}</span></div>
      <div className="ess-track">
        <div className={`ess-fill${e < threshold ? " low" : ""}`} style={{ width: `${(e / n) * 100}%` }} />
        <div className="ess-threshold" style={{ left: `${(threshold / n) * 100}%` }} title="resampling threshold" />
      </div>
      {!compact && <div className="fine">1/Σw². {n} when weights are even, 1 when one particle holds everything. Below the marker ({threshold}) the cloud is resampled.</div>}
    </div>
  );
}

function ResampleDetail({ step }: { step: Step }) {
  const anc: number[] = step.detail.ancestors;
  const n = anc.length;
  const H = 18 * n + 8;
  const counts = new Map<number, number>();
  anc.forEach((a) => counts.set(a, (counts.get(a) ?? 0) + 1));
  return (
    <div className="detail">
      <svg viewBox={`0 0 260 ${H}`} className="ancestry" role="img" aria-label="resampling ancestry">
        <text x={20} y={10} className="fine-svg">before</text>
        <text x={240} y={10} className="fine-svg" textAnchor="end">after</text>
        {anc.map((a, k) => (
          <path key={k} d={`M40,${22 + a * 18} C130,${22 + a * 18} 130,${22 + k * 18} 220,${22 + k * 18}`} className="anc-line" />
        ))}
        {Array.from({ length: n }, (_, k) => (
          <g key={k}>
            <circle cx={40} cy={22 + k * 18} r={6} className={`anc-dot${counts.has(k) ? "" : " dead"}`} />
            <text x={28} y={26 + k * 18} textAnchor="end" className="fine-svg">P{k + 1}</text>
            <circle cx={220} cy={22 + k * 18} r={6} className="anc-dot" />
            <text x={232} y={26 + k * 18} className="fine-svg">P{k + 1}</text>
          </g>
        ))}
      </svg>
      <div className="fine">Hollow = no descendants. Every new particle is a full copy of its ancestor's database; weights reset to equal.</div>
    </div>
  );
}

function RejuvDetail({ step, palette }: { step: Step; palette: Map<string, string> }) {
  const d = step.detail;
  const settings: Setting[] = d.settings;
  const order = settings.map((_, i) => i).sort((a, b) => settings[b].p - settings[a].p);
  const top = order.slice(0, 5);
  const best = order[0];
  const alt = order.find((i) => i !== best);
  const label = (s: Setting) => Object.entries(s.values).map(([f, v]) => (d.targets.length > 1 ? `${f}=${show(v)}` : show(v))).join(", ");
  const nameOf = (s: Setting) => {
    const v = Object.values(s.values)[0];
    return typeof v === "string" && !v.includes("#") ? v : null;
  };
  return (
    <div className="detail">
      <div className="choice">
        <div className="choice-head">
          <span className="pill rejuv">blocked Gibbs</span>
          <span className="mono">{d.key}</span> · redraw <span className="mono">{d.targets.join(", ")}</span>
        </div>
        {top.map((i) => (
          <Bar key={i} p={settings[i].p} chosen={i === d.chosen} label={label(settings[i])}
            sub={Object.values(settings[i].values).includes(Object.values(d.old)[0] as Value) ? "current" : null}
            color={colorFor(palette, nameOf(settings[i]))} />
        ))}
        {settings.length > top.length && <div className="fine">+{settings.length - top.length} more with ≈0 probability</div>}
      </div>
      {alt !== undefined && (
        <div className="evidence">
          <div className="evidence-title">Why: evidence, counted once per object</div>
          <table>
            <thead>
              <tr><th></th><th className="mono">{label(settings[best])}</th><th className="mono">{label(settings[alt])}</th></tr>
            </thead>
            <tbody>
              <tr><td>log prior</td><td className="mono">{f2(settings[best].prior)}</td><td className="mono">{f2(settings[alt].prior)}</td></tr>
              {evidenceRows(settings[best], settings[alt]).map(([k, a, b]) => (
                <tr key={k}><td className="mono">{k}</td><td className="mono">{f2(a)}</td><td className="mono">{f2(b)}</td></tr>
              ))}
              <tr className="total"><td>total</td><td className="mono">{f2(settings[best].score)}</td><td className="mono">{f2(settings[alt].score)}</td></tr>
            </tbody>
          </table>
          <div className="fine">Each referring object contributes its observed cells once, however many rows mention it. Counting per row would rebuild majority voting inside the model.</div>
        </div>
      )}
    </div>
  );
}

function evidenceRows(a: Setting, b: Setting): [string, number | null, number | null][] {
  const keys = new Set([...a.evidence.map((e) => e.key), ...b.evidence.map((e) => e.key)]);
  const get = (s: Setting, k: string) => s.evidence.find((e) => e.key === k)?.logp ?? 0;
  return [...keys].sort().map((k) => [k, get(a, k), get(b, k)]);
}

function RowDetail({ rec, step }: { rec: Recording; step: Step }) {
  const row = rec.rows[step.row!];
  return (
    <div className="detail">
      <table className="kvtable">
        <tbody>
          {Object.entries(row.data).map(([k, v]) => (
            <tr key={k}><td className="mono">{k}</td><td className="mono">"{show(v)}"</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DoneDetail({ rec, step, state, palette }: { rec: Recording; step: Step; state: StepState; palette: Map<string, string> }) {
  // Posterior over each row's clean value: every particle votes with its final weight.
  const w = normWeights(step.logw);
  const rows = rec.rows.map((r, i) => {
    const counts = new Map<string, number>();
    state.dbs.forEach((db, k) => {
      const v = show(cleanValue(rec, db, i));
      counts.set(v, (counts.get(v) ?? 0) + w[k]);
    });
    return { r, i, counts };
  });
  const seen = new Set<string>();
  const uniq = rows.filter(({ r }) => {
    const k = JSON.stringify(r.data);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return (
    <div className="detail">
      <div className="fine">Distinct rows, with the posterior over each one's clean value (particles weighted by their final weights):</div>
      <table className="kvtable">
        <tbody>
          {uniq.map(({ r, i, counts }) => (
            <tr key={i}>
              <td className="mono">{Object.values(r.data).map(show).join(" · ")}</td>
              <td>
                {[...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v, p]) => (
                  <span key={v} className="post-chip" style={{ borderColor: colorFor(palette, v) }}>
                    {v} <b>{Math.round(p * 100)}%</b>
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StepDetail(props: {
  rec: Recording; step: Step; state: StepState; focus: number; onFocus: (k: number) => void; palette: Map<string, string>;
}) {
  const { step } = props;
  switch (step.kind) {
    case "propose": return <ProposeDetail step={step} palette={props.palette} />;
    case "reweight": return <WeightBars step={step} state={props.state} focus={props.focus} onFocus={props.onFocus} />;
    case "resample": return <ResampleDetail step={step} />;
    case "rejuv": return <RejuvDetail step={step} palette={props.palette} />;
    case "row": return <RowDetail rec={props.rec} step={step} />;
    case "done": return <DoneDetail rec={props.rec} step={step} state={props.state} palette={props.palette} />;
    default: return null;
  }
}

export function ModelView({ rec }: { rec: Recording }) {
  const { classes, observation_class } = rec.schema;
  const obs = classes.find((c) => c.name === observation_class)!;
  return (
    <div className="model">
      <div className="model-classes">
        {classes.map((c) => (
          <div key={c.name} className="model-class">
            <div className="model-class-name">{c.name}</div>
            {c.fields.map((f) => (
              <div key={f.name} className="model-field">
                <span className={`glyph ${f.kind === "slot" ? "slot" : f.observed ? "observed" : "latent"}${f.key ? " key" : ""}`} />
                <span className="mono">{f.name}</span>
                <span className="model-dist">
                  {f.kind === "slot" ? `~ ${f.target}` : `~ ${f.dist}(${f.parents.join(", ")})`}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="fine">
        After flattening, <span className="mono">{observation_class}</span>'s own graph holds{" "}
        {obs.flattened.map((v, i) => <span key={v}>{i ? ", " : ""}<span className="mono">{v}</span></span>)}
        {" "}— one enumeration covers the slot choices and the evidence bearing on them.
      </div>
    </div>
  );
}
