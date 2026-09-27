import { useMemo, type CSSProperties, type JSX } from "react";
import type { Db, Obj, Recording, Value } from "./types";
import { CARD_W, FIELD_H, HEADER_H, fieldAnchor, layout, type Box, type Layout } from "./layout";
import { colorFor, rootName } from "./colors";

export interface Highlight {
  added: Set<string>;
  changed: Map<string, Set<string>>; // key -> fields
  focus: Set<string>; // objects the step is about
  ghosts: Db; // objects deleted by this step, drawn where they were
}

export const NO_HIGHLIGHT: Highlight = { added: new Set(), changed: new Map(), focus: new Set(), ghosts: {} };

interface Props {
  rec: Recording;
  db: Db;
  prevDb?: Db;
  palette: Map<string, string>;
  hl?: Highlight;
  compact?: boolean;
  stepKey?: number; // restarts flash animations each step
  onHover?: (key: string | null) => void;
}

function curve(x1: number, y1: number, x2: number, y2: number) {
  const my = (y1 + y2) / 2;
  return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
}

function fmt(v: Value): string {
  if (v === null || v === undefined) return "·";
  const s = String(v);
  return s.length > 13 ? s.slice(0, 12) + "…" : s;
}

export function NetView({ rec, db, prevDb, palette, hl = NO_HIGHLIGHT, compact, stepKey, onHover }: Props) {
  const schema = rec.schema;
  // Lay out the union of current objects and this step's ghosts, so a collected object
  // is drawn in place for the step that removes it.
  const lay: Layout = useMemo(() => {
    const all = Object.keys(hl.ghosts).length ? { ...hl.ghosts, ...db } : db;
    return layout(schema, all);
  }, [schema, db, hl.ghosts]);

  const all: Db = useMemo(() => ({ ...hl.ghosts, ...db }), [db, hl.ghosts]);
  const color = (o: Obj) => colorFor(palette, rootName(schema, all, o));

  const slotEdges: JSX.Element[] = [];
  const depEdges: JSX.Element[] = [];
  for (const o of Object.values(all)) {
    const k = `${o.cls}#${o.oid}`;
    const box = lay.boxes[k];
    if (!box) continue;
    const cls = schema.classes.find((c) => c.name === o.cls)!;
    for (const f of cls.fields) {
      if (f.kind === "slot") {
        const v = o.values[f.name];
        if (v === null || v === undefined) continue;
        const tk = `${f.target}#${v}`;
        const tb = lay.boxes[tk];
        if (!tb) continue;
        const a = box.chip ? { x: box.x + box.w / 2, y: box.y } : { x: box.x + box.w / 2, y: box.y };
        const b = { x: tb.x + tb.w / 2, y: tb.y + tb.h };
        const d = curve(a.x, a.y, b.x, b.y);
        const moved = hl.changed.get(k)?.has(f.name);
        slotEdges.push(
          <path key={`s-${k}-${f.name}`} className={`edge slot${moved ? " moved" : ""}${hl.ghosts[k] ? " ghost" : ""}`}
            d={d} style={{ d: `path('${d}')` } as CSSProperties} />,
        );
      } else if (!compact) {
        for (const p of f.parents) {
          const parts = p.split(".");
          let src: Obj | undefined = o;
          let ok = true;
          for (let i = 0; i < parts.length - 1; i++) {
            const sc = schema.classes.find((c) => c.name === src!.cls)!;
            const sf = sc.fields.find((x) => x.name === parts[i]);
            const sv: Value = src!.values[parts[i]];
            if (!sf || sv === null || sv === undefined) { ok = false; break; }
            src = all[`${sf.target}#${sv}`];
            if (!src) { ok = false; break; }
          }
          if (!ok || !src) continue;
          const sb = lay.boxes[`${src.cls}#${src.oid}`];
          if (!sb) continue;
          const a = fieldAnchor(schema, sb, parts[parts.length - 1]);
          const b = fieldAnchor(schema, box, f.name);
          // Leave the parent card on its right edge and enter the child from the right,
          // so dependency arrows do not run under the slot edges.
          const x1 = sb.chip ? a.x : sb.x + sb.w - 4;
          const x2 = box.chip ? b.x : box.x + box.w - 4;
          const d = `M${x1},${a.y} C${x1 + 40},${a.y} ${x2 + 40},${b.y} ${x2 + 2},${b.y}`;
          depEdges.push(
            <path key={`d-${k}-${f.name}-${p}`} className="edge dep" d={d}
              style={{ d: `path('${d}')` } as CSSProperties} markerEnd="url(#arrow)" />,
          );
        }
      }
    }
  }

  const pad = compact ? 4 : 0;
  return (
    <svg className={`net${compact ? " compact" : ""}`} viewBox={`${-pad} 0 ${lay.width + 2 * pad} ${lay.height}`}
      preserveAspectRatio="xMidYMin meet" role="img"
      // Never shrink below ~80% of natural size: past that the labels stop being legible,
      // and a sideways scroll is the better trade.
      style={compact ? undefined : { minWidth: Math.round(lay.width * 0.8), maxWidth: Math.round(lay.width * 1.1) }}
      aria-label={compact ? "particle thumbnail" : "unrolled Bayes net of the latent database"}>
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L8,4 L0,8 z" className="arrowhead" />
        </marker>
      </defs>
      {!compact && lay.lanes.map((l) => (
        <g key={l.cls}>
          <line className="lane-rule" x1={0} x2={lay.width} y1={l.y - 10} y2={l.y - 10} />
          <text className="lane-label" x={4} y={l.y - 14}>{l.cls}</text>
        </g>
      ))}
      <g>{slotEdges}</g>
      <g>{depEdges}</g>
      {Object.values(all).map((o) => {
        const k = `${o.cls}#${o.oid}`;
        const box = lay.boxes[k];
        if (!box) return null;
        return (
          <ObjCard key={k} rec={rec} o={o} box={box} color={color(o)} compact={compact}
            added={hl.added.has(k)} focus={hl.focus.has(k)} ghost={!!hl.ghosts[k]}
            changed={hl.changed.get(k)} stepKey={stepKey} prevDb={prevDb}
            onHover={onHover} />
        );
      })}
    </svg>
  );
}

interface CardProps {
  rec: Recording;
  o: Obj;
  box: Box;
  color: string;
  compact?: boolean;
  added: boolean;
  focus: boolean;
  ghost: boolean;
  changed?: Set<string>;
  stepKey?: number;
  prevDb?: Db;
  onHover?: (key: string | null) => void;
}

function ObjCard({ rec, o, box, color, compact, added, focus, ghost, changed, stepKey, onHover }: CardProps) {
  const cls = rec.schema.classes.find((c) => c.name === o.cls)!;
  const k = `${o.cls}#${o.oid}`;
  const state = `${added ? " added" : ""}${focus ? " focus" : ""}${ghost ? " ghost" : ""}`;
  const style = { transform: `translate(${box.x}px, ${box.y}px)` } as CSSProperties;

  if (box.chip) {
    const attr = cls.fields.find((f) => f.kind === "attr");
    return (
      <g className={`obj chip${state}`} style={style} onMouseEnter={() => onHover?.(k)} onMouseLeave={() => onHover?.(null)}>
        <rect className="chip-bg" width={box.w} height={box.h} rx={5}
          style={compact ? { stroke: color, fill: color } : { stroke: color }} />
        <rect width={4} height={box.h} rx={2} style={{ fill: color }} />
        {!compact && (
          <>
            <text className="chip-label" x={9} y={15}>r{o.oid + 1}</text>
            {attr && (
              <>
                <circle cx={40} cy={11} r={4} className={`var${attr.observed ? " observed" : ""}`} />
                <text className="chip-value" x={48} y={15}>{fmt(o.values[attr.name])}</text>
              </>
            )}
          </>
        )}
        <title>{`${k}\n${cls.fields.map((f) => `${f.name} = ${o.values[f.name] ?? "·"}`).join("\n")}`}</title>
      </g>
    );
  }

  return (
    <g className={`obj card${state}`} style={style} onMouseEnter={() => onHover?.(k)} onMouseLeave={() => onHover?.(null)}>
      <rect className="card-bg" width={CARD_W} height={box.h} rx={8}
        style={compact ? { fill: color, stroke: color } : undefined} />
      <rect className="card-stripe" width={CARD_W} height={4} rx={2} style={{ fill: color }} />
      {!compact && (
        <>
          <text className="card-title" x={10} y={17}>{o.cls} <tspan className="card-id">#{o.oid}</tspan></text>
          <text className="card-refs" x={CARD_W - 8} y={17} textAnchor="end">
            <title>How many reference slots point here: the CRP count n_r that drives rich-get-richer.</title>
            n<tspan dy={2} fontSize={8}>r</tspan><tspan dy={-2}>={o.refs}</tspan>
          </text>
          {cls.fields.map((f, i) => {
            const y = HEADER_H + i * FIELD_H;
            const v = o.values[f.name];
            const isChanged = changed?.has(f.name);
            return (
              <g key={f.name} transform={`translate(0, ${y})`}>
                {isChanged && <rect key={`flash-${stepKey}`} className="flash" x={3} y={1} width={CARD_W - 6} height={FIELD_H - 2} rx={4} />}
                {f.kind === "slot" ? (
                  <rect x={8} y={FIELD_H / 2 - 4.5} width={9} height={9} rx={1.5} className="var slot" />
                ) : (
                  <>
                    <circle cx={13} cy={FIELD_H / 2} r={5} className={`var${f.observed ? " observed" : ""}`} />
                    {f.key && <circle cx={13} cy={FIELD_H / 2} r={7.5} className="var keyring" />}
                  </>
                )}
                <text className="field-name" x={24} y={FIELD_H / 2 + 4}>{f.name}</text>
                <text className={`field-value${f.kind === "slot" ? " slot" : ""}`} x={CARD_W - 8} y={FIELD_H / 2 + 4} textAnchor="end">
                  {f.kind === "slot" ? (v === null || v === undefined ? "·" : `→ ${f.target} #${v}`) : fmt(v)}
                </text>
              </g>
            );
          })}
        </>
      )}
      <title>{`${k}  (n_r = ${o.refs})\n${cls.fields.map((f) => `${f.name} = ${o.values[f.name] ?? "·"}${f.observed ? "  [observed]" : ""}`).join("\n")}`}</title>
    </g>
  );
}
