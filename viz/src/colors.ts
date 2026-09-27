import type { Db, Obj, Recording, Schema, Value } from "./types";

// Categorical palette (Okabe–Ito, reordered), readable on light and dark surfaces.
const PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#8C6D1F", "#7A5195"];
export const NEUTRAL = "var(--muted)";

// Every value the top class's first attribute ever takes (city names) gets a stable
// colour, so the same hypothesis is the same colour in every particle and every step.
export function buildPalette(rec: Recording): Map<string, string> {
  const top = rec.schema.classes[0];
  const attr = top.fields.find((f) => f.kind === "attr")?.name;
  const names = new Set<string>();
  for (const s of rec.steps)
    for (const op of s.ops) {
      if (op.op === "add" && op.obj.cls === top.name && attr) {
        const v = op.obj.values[attr];
        if (typeof v === "string") names.add(v);
      }
      if (op.op === "set" && op.key.startsWith(top.name + "#") && op.field === attr && typeof op.value === "string")
        names.add(op.value);
    }
  const sorted = [...names].sort();
  return new Map(sorted.map((n, i) => [n, PALETTE[i % PALETTE.length]]));
}

// The top-class attribute an object ultimately hangs off: a Record's city name.
export function rootName(schema: Schema, db: Db, o: Obj | undefined): Value {
  let cur = o;
  for (let guard = 0; cur && guard < 8; guard++) {
    const cls = schema.classes.find((c) => c.name === cur!.cls)!;
    if (cls === schema.classes[0]) {
      const attr = cls.fields.find((f) => f.kind === "attr");
      return attr ? cur.values[attr.name] ?? null : null;
    }
    const slot = cls.fields.find((f) => f.kind === "slot");
    if (!slot) return null;
    const v = cur.values[slot.name];
    cur = v === null || v === undefined ? undefined : db[`${slot.target}#${v}`];
  }
  return null;
}

export function colorFor(palette: Map<string, string>, name: Value): string {
  return typeof name === "string" ? palette.get(name) ?? NEUTRAL : NEUTRAL;
}
