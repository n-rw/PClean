import type { Db, Obj, Recording, StepState, Value } from "./types";

// Replay every step's ops once, up front, into an array of immutable states. Each step
// copies only the particles and objects it touches, so 500 steps x 6 particles is cheap
// and scrubbing anywhere on the timeline is an array index.
export function replay(rec: Recording): StepState[] {
  const n = rec.n_particles;
  let cur: StepState = {
    dbs: Array.from({ length: n }, () => ({})),
    lineage: Array.from({ length: n }, (_, k) => k),
  };
  const out: StepState[] = [];
  for (const step of rec.steps) {
    let dbs = cur.dbs.slice();
    let lineage = cur.lineage;
    const touched = new Set<number>();
    const own = (p: number): Db => {
      if (!touched.has(p)) {
        dbs[p] = { ...dbs[p] };
        touched.add(p);
      }
      return dbs[p];
    };
    for (const op of step.ops) {
      switch (op.op) {
        case "resample":
          // Clones share structure until something writes to them, which `own` handles.
          dbs = op.ancestors.map((a) => cur.dbs[a]);
          lineage = op.ancestors.map((a) => cur.lineage[a]);
          touched.clear();
          break;
        case "add":
          own(op.p)[op.key] = op.obj;
          break;
        case "set": {
          const db = own(op.p);
          const o = db[op.key];
          db[op.key] = { ...o, values: { ...o.values, [op.field]: op.value } };
          break;
        }
        case "refs": {
          const db = own(op.p);
          db[op.key] = { ...db[op.key], refs: op.value };
          break;
        }
        case "del":
          delete own(op.p)[op.key];
          break;
      }
    }
    cur = { dbs, lineage };
    out.push(cur);
  }
  return out;
}

// Follow a dotted path of reference slots from an object: "practice.city.name".
export function resolve(
  rec: Recording,
  db: Db,
  from: Obj,
  path: string,
): { obj: Obj; field: string; value: Value } | null {
  const parts = path.split(".");
  let obj: Obj | undefined = from;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!obj) return null;
    const cls = rec.schema.classes.find((c) => c.name === obj!.cls);
    const f = cls?.fields.find((x) => x.name === parts[i]);
    if (!f || f.kind !== "slot") return null;
    const target: Value = obj.values[parts[i]];
    if (target === null || target === undefined) return null;
    obj = db[`${f.target}#${target}`];
  }
  if (!obj) return null;
  const field = parts[parts.length - 1];
  return { obj, field, value: obj.values[field] ?? null };
}

export function cleanValue(rec: Recording & { clean_path?: string }, db: Db, row: number): Value {
  const r = db[`${rec.schema.observation_class}#${row}`];
  if (!r || !rec.clean_path) return null;
  return resolve(rec, db, r, rec.clean_path)?.value ?? null;
}

export function normWeights(logw: (number | null)[]): number[] {
  const finite = logw.filter((w): w is number => w !== null);
  if (!finite.length) return logw.map(() => 0);
  const m = Math.max(...finite);
  const ex = logw.map((w) => (w === null ? 0 : Math.exp(w - m)));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / z);
}
