// Shapes of the event log written by python/viz_record.py.

export type Value = string | number | null;

export interface Field {
  name: string;
  kind: "slot" | "attr";
  observed: boolean;
  key: boolean;
  target?: string;
  dist?: string;
  parents: string[]; // dotted paths from the owning object, e.g. "city.name"
}

export interface ClassSchema {
  name: string;
  fields: Field[];
  flattened: string[];
}

export interface Schema {
  classes: ClassSchema[]; // topological order: parents first
  observation_class: string;
}

export interface Obj {
  cls: string;
  oid: number;
  values: Record<string, Value>;
  refs: number;
}

export type Db = Record<string, Obj>; // "City#0" -> object

export type Op =
  | { op: "add"; p: number; key: string; obj: Obj }
  | { op: "set"; p: number; key: string; field: string; value: Value; old: Value }
  | { op: "refs"; p: number; key: string; value: number }
  | { op: "del"; p: number; key: string }
  | { op: "resample"; ancestors: number[] };

export interface Candidate {
  label: Value;
  summary: string | null;
  prior: number | null;
  score: number | null;
  p: number;
}

export type Choice =
  | { kind: "slot" | "attribute"; vertex: string; target: string | null;
      candidates: Candidate[]; chosen: number | null }
  | { kind: "key_lookup"; vertex: string; target: string; key: Value[]; hit: string | null }
  | { kind: "observed"; vertex: string; value: Value; logp: number | null; args: Value[] }
  | { kind: "determined"; vertex: string; value: Value }
  | { kind: "sampled"; vertex: string; value: Value; logp: number | null };

export interface Setting {
  values: Record<string, Value>;
  prior: number | null;
  score: number | null;
  p: number;
  evidence: { key: string; logp: number | null }[];
}

export type StepKind =
  | "intro" | "row" | "propose" | "reweight" | "resample"
  | "sweep" | "rejuv" | "gc" | "done";

export interface Step {
  i: number;
  kind: StepKind;
  phase: "model" | "smc" | "resample" | "rejuv" | "done";
  title: string;
  text: string;
  particle: number | null;
  row: number | null;
  sweep: number | null;
  ops: Op[];
  detail: any;
  logw: (number | null)[];
  ess: number | null;
}

export interface RowInfo {
  data: Record<string, Value>;
  observed: string;
  truth: string | null;
}

export interface Recording {
  id: string;
  title: string;
  blurb: string;
  seed: number;
  n_particles: number;
  schema: Schema;
  clean_path?: string;
  rows: RowInfo[];
  stats: Record<string, unknown>;
  steps: Step[];
}

export interface StepState {
  dbs: Db[];
  // For each particle, which original particle it descends from (after resampling).
  lineage: number[];
}
