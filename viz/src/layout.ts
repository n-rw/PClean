import type { ClassSchema, Db, Obj, Schema } from "./types";

// Deterministic lane layout for one particle's unrolled net.
//
// One horizontal lane per class, parents on top (the schema's topological order), so
// every Bayes-net arrow points downward. Each object sits centred over the objects that
// refer to it, and the observation class is drawn as compact chips grouped under their
// parent. Order is by object id within a group, so adding an object never reshuffles
// the ones already placed -- the net visibly *grows* rather than re-flowing.

export const CARD_W = 164;
export const HEADER_H = 22;
export const FIELD_H = 19;
export const CHIP_H = 22;
const GAP_X = 18;
const GAP_Y = 64;
const CHIP_GAP = 6;
const PAD = 16;

export interface Box {
  key: string;
  cls: string;
  x: number;
  y: number;
  w: number;
  h: number;
  chip: boolean;
}

export interface Layout {
  boxes: Record<string, Box>;
  width: number;
  height: number;
  lanes: { cls: string; y: number; h: number }[];
}

export function chipWidth(schema: Schema): number {
  const obs = schema.classes.find((c) => c.name === schema.observation_class)!;
  return obs.fields.some((f) => f.kind === "attr") ? 108 : 50;
}

function cardHeight(cls: ClassSchema) {
  return HEADER_H + cls.fields.length * FIELD_H + 6;
}

function primaryParent(schema: Schema, o: Obj): string | null {
  const cls = schema.classes.find((c) => c.name === o.cls)!;
  const slot = cls.fields.find((f) => f.kind === "slot");
  if (!slot) return null;
  const v = o.values[slot.name];
  return v === null || v === undefined ? null : `${slot.target}#${v}`;
}

export function layout(schema: Schema, db: Db): Layout {
  const lanesCls = schema.classes;
  const obsName = schema.observation_class;
  const chipW = chipWidth(schema);
  const byLane: Record<string, Obj[]> = {};
  for (const c of lanesCls) byLane[c.name] = [];
  for (const o of Object.values(db)) byLane[o.cls]?.push(o);
  for (const c of lanesCls) byLane[c.name].sort((a, b) => a.oid - b.oid);

  const children: Record<string, Obj[]> = {};
  const orphans: Record<string, Obj[]> = {};
  for (const c of lanesCls) orphans[c.name] = [];
  lanesCls.forEach((c, li) => {
    for (const o of byLane[c.name]) {
      const p = li === 0 ? null : primaryParent(schema, o);
      if (p && db[p]) (children[p] ??= []).push(o);
      else if (li > 0) orphans[c.name].push(o);
    }
  });

  const key = (o: Obj) => `${o.cls}#${o.oid}`;
  const chipCols = (n: number) => Math.max(1, Math.min(n, Math.floor((CARD_W + CHIP_GAP) / (chipW + CHIP_GAP)) || 1));

  // Width of the subtree rooted at o (object plus everything grouped under it).
  const widthMemo: Record<string, number> = {};
  function subtreeWidth(o: Obj): number {
    const k = key(o);
    if (widthMemo[k] !== undefined) return widthMemo[k];
    if (o.cls === obsName) return (widthMemo[k] = chipW);
    const kids = children[k] ?? [];
    let w = CARD_W;
    if (kids.length) {
      if (kids[0].cls === obsName) {
        const cols = chipCols(kids.length);
        w = Math.max(w, cols * chipW + (cols - 1) * CHIP_GAP);
      } else {
        const sum = kids.reduce((a, c) => a + subtreeWidth(c), 0) + GAP_X * (kids.length - 1);
        w = Math.max(w, sum);
      }
    }
    return (widthMemo[k] = w);
  }

  // Lane geometry.
  const lanes: Layout["lanes"] = [];
  let y = PAD + 14;
  const laneY: Record<string, number> = {};
  for (const c of lanesCls) {
    let h: number;
    if (c.name === obsName) {
      let rows = 1;
      for (const [pk, kids] of Object.entries(children)) {
        if (kids[0]?.cls === obsName && db[pk]) {
          rows = Math.max(rows, Math.ceil(kids.length / chipCols(kids.length)));
        }
      }
      h = rows * (CHIP_H + CHIP_GAP);
    } else {
      h = cardHeight(c);
    }
    laneY[c.name] = y;
    lanes.push({ cls: c.name, y, h });
    y += h + GAP_Y;
  }

  const boxes: Record<string, Box> = {};
  function place(o: Obj, x0: number) {
    const k = key(o);
    const w = subtreeWidth(o);
    const cls = lanesCls.find((c) => c.name === o.cls)!;
    if (o.cls === obsName) {
      boxes[k] = { key: k, cls: o.cls, x: x0, y: laneY[o.cls], w: chipW, h: CHIP_H, chip: true };
      return;
    }
    boxes[k] = { key: k, cls: o.cls, x: x0 + (w - CARD_W) / 2, y: laneY[o.cls], w: CARD_W, h: cardHeight(cls), chip: false };
    const kids = children[k] ?? [];
    if (!kids.length) return;
    if (kids[0].cls === obsName) {
      const cols = chipCols(kids.length);
      const gw = cols * chipW + (cols - 1) * CHIP_GAP;
      const gx = x0 + (w - gw) / 2;
      kids.forEach((kid, i) => {
        const kk = key(kid);
        boxes[kk] = {
          key: kk, cls: kid.cls,
          x: gx + (i % cols) * (chipW + CHIP_GAP),
          y: laneY[kid.cls] + Math.floor(i / cols) * (CHIP_H + CHIP_GAP),
          w: chipW, h: CHIP_H, chip: true,
        };
      });
    } else {
      const sum = kids.reduce((a, c) => a + subtreeWidth(c), 0) + GAP_X * (kids.length - 1);
      let cx = x0 + (w - sum) / 2;
      for (const kid of kids) {
        place(kid, cx);
        cx += subtreeWidth(kid) + GAP_X;
      }
    }
  }

  let x = PAD;
  for (const o of byLane[lanesCls[0].name]) {
    place(o, x);
    x += subtreeWidth(o) + GAP_X * 2;
  }
  // Objects whose parent is missing (e.g. a Practice pointing at a collected City) go at
  // the end of their lane, still visible.
  for (const c of lanesCls.slice(1)) {
    for (const o of orphans[c.name]) {
      place(o, x);
      x += subtreeWidth(o) + GAP_X;
    }
  }

  return { boxes, width: Math.max(x + PAD, 480), height: y - GAP_Y + PAD + 8, lanes };
}

export function fieldAnchor(schema: Schema, box: Box, field: string): { x: number; y: number } {
  if (box.chip) return { x: box.x + box.w / 2, y: box.y + box.h / 2 };
  const cls = schema.classes.find((c) => c.name === box.cls)!;
  const i = cls.fields.findIndex((f) => f.name === field);
  return { x: box.x + 13, y: box.y + HEADER_H + i * FIELD_H + FIELD_H / 2 };
}
