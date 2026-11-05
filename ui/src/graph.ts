// Workflow <-> node graph: steps are nodes, clip references are edges. The workflow stays the
// source of truth (steps in an order where every clip is made before it is read).

import type { Step, Workflow, WorkflowOp } from "./api";

export interface StepUi {
  id: string;
  x: number;
  y: number;
}

let counter = 0;
const newId = () => `n${Date.now().toString(36)}${(counter++).toString(36)}`;

export const clipFields = (op?: WorkflowOp) => (op ? op.fields.filter((f) => f.type === "clip" || f.type === "clips") : []);
export const outName = (s: Step) => String(s.out ?? s.op);

// References of step i: [field, clip name, producer index].
export function references(steps: Step[], ops: Map<string, WorkflowOp>, i: number): { field: string; clip: string; from: number; slot: number }[] {
  const out: { field: string; clip: string; from: number; slot: number }[] = [];
  const s = steps[i];
  for (const f of clipFields(ops.get(s.op))) {
    const v = s[f.name];
    const names = Array.isArray(v) ? (v as unknown[]).map(String) : typeof v === "string" ? [v] : [];
    names.forEach((name, slot) => {
      for (let j = i - 1; j >= 0; --j)
        if (ops.get(steps[j].op)?.output && outName(steps[j]) === name) {
          out.push({ field: f.name, clip: name, from: j, slot });
          break;
        }
    });
  }
  return out;
}

// Every step gets an id and a position; missing positions are laid out by depth (columns) in order.
export function withLayout(wf: Workflow, ops: Map<string, WorkflowOp>): Workflow {
  const steps = wf.steps;
  if (steps.every((s) => s.ui && typeof (s.ui as Partial<StepUi>).id === "string" && typeof s.ui.x === "number")) return wf;
  const depth: number[] = [];
  const rows = new Map<number, number>();
  const next = steps.map((s, i) => {
    const refs = references(steps, ops, i);
    depth[i] = refs.length ? Math.max(...refs.map((r) => depth[r.from] + 1)) : 0;
    const row = rows.get(depth[i]) ?? 0;
    rows.set(depth[i], row + 1);
    const ui = s.ui as Partial<StepUi> | undefined;
    return {
      ...s,
      ui: {
        id: typeof ui?.id === "string" ? ui.id : newId(),
        x: typeof ui?.x === "number" ? ui.x : 40 + depth[i] * 300,
        y: typeof ui?.y === "number" ? ui.y : 40 + row * 170,
      },
    } as Step;
  });
  return { ...wf, steps: next };
}

export const stepId = (s: Step) => String((s.ui as Partial<StepUi> | undefined)?.id ?? "");

// Steps reordered so producers come before readers (stable); null when the references form a cycle.
export function topoSort(steps: Step[], ops: Map<string, WorkflowOp>): Step[] | null {
  // Resolve references by name against any producer (not only earlier ones), so new edges can point backwards.
  const producers = new Map<string, number[]>();
  steps.forEach((s, i) => {
    if (ops.get(s.op)?.output) producers.set(outName(s), [...(producers.get(outName(s)) ?? []), i]);
  });
  const deps = steps.map((s, i) => {
    const d = new Set<number>();
    for (const f of clipFields(ops.get(s.op))) {
      const v = s[f.name];
      const names = Array.isArray(v) ? (v as unknown[]).map(String) : typeof v === "string" ? [v] : [];
      for (const n of names) {
        const ps = (producers.get(n) ?? []).filter((p) => p !== i);
        if (!ps.length) continue;
        // The nearest earlier producer if there is one, else the first later one.
        const earlier = ps.filter((p) => p < i);
        d.add(earlier.length ? earlier[earlier.length - 1] : ps[0]);
      }
    }
    return d;
  });
  const done = new Array(steps.length).fill(false);
  const order: number[] = [];
  while (order.length < steps.length) {
    const i = steps.findIndex((_, k) => !done[k] && [...deps[k]].every((d) => done[d]));
    if (i < 0) return null;
    done[i] = true;
    order.push(i);
  }
  return order.map((i) => steps[i]);
}

// A clip name not used by any step yet.
export function uniqueName(steps: Step[], base: string): string {
  const used = new Set(steps.map(outName));
  if (!used.has(base)) return base;
  let k = 2;
  while (used.has(`${base}${k}`)) ++k;
  return `${base}${k}`;
}

export const makeStepUi = (x: number, y: number): StepUi => ({ id: newId(), x, y });

// A position near (x, y) that does not cover an existing node (nodes are ~240 x 150).
export function freeSpot(steps: Step[], x: number, y: number): StepUi {
  const taken = (py: number) =>
    steps.some((s) => {
      const u = s.ui as Partial<StepUi> | undefined;
      return typeof u?.x === "number" && typeof u.y === "number" && Math.abs(u.x - x) < 250 && Math.abs(u.y - py) < 150;
    });
  while (taken(y)) y += 170;
  return makeStepUi(x, y);
}
