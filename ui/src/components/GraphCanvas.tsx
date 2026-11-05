import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { memo, useEffect, useMemo, useState } from "react";
import { frameUrl, type ModelInfo, type Step, type WorkflowOp } from "../api";
import type { LiveJob } from "../engine";
import { clipFields, outName, references, stepId, topoSort, uniqueName, type StepUi } from "../graph";
import { fieldLabel } from "./FieldInput";

interface StepData extends Record<string, unknown> {
  step: Step;
  op?: WorkflowOp;
  index: number;
  status: "" | "running" | "done" | "failed";
  stage: string;
  thumb?: string;
  frames?: number;
  invalid: boolean;
}

const summary = (s: Step): string => {
  const parts: string[] = [];
  if (typeof s.prompt === "string" && s.prompt) parts.push(`“${s.prompt.length > 42 ? s.prompt.slice(0, 40) + "…" : s.prompt}”`);
  if (typeof s.model === "string") parts.push(s.model.split("/").pop() ?? "");
  for (const k of ["frames", "steps", "width", "passes", "path"]) if (s[k] !== undefined) parts.push(`${fieldLabel(k).toLowerCase()} ${String(s[k])}`);
  return parts.slice(0, 3).join(" · ");
};

const StepNode = memo(function StepNode({ data, selected }: NodeProps<Node<StepData>>) {
  const { step, op, status, stage, thumb, frames, invalid } = data;
  const inputs = clipFields(op);
  const border = status === "running" ? "var(--run)" : selected ? "var(--accent)" : invalid ? "var(--bad)" : "var(--line2)";
  return (
    <div className="node" style={{ borderColor: border, background: selected ? "var(--accent-bg)" : "var(--panel)" }}>
      <div className="node-head">
        <strong>{op?.title ?? step.op}</strong>
        <span className="help">{step.op}</span>
      </div>
      {inputs.map((f, k) => (
        <div key={f.name} className="node-port in" style={{ top: 44 + k * 22 }}>
          <Handle type="target" position={Position.Left} id={f.name} style={{ top: "50%" }} />
          <span>{fieldLabel(f.name)}{f.type === "clips" ? " …" : ""}{f.required ? "" : "?"}</span>
        </div>
      ))}
      <div className="node-body" style={{ paddingTop: inputs.length ? inputs.length * 22 + 4 : 0 }}>
        <span className="note">{summary(step) || op?.doc}</span>
        {thumb ? (
          <div className="node-thumb">
            <img src={thumb} alt="" loading="lazy" />
            {frames && frames > 1 ? <span>{frames} kare</span> : null}
          </div>
        ) : null}
        {status ? <span className={`state ${status}`}>{status === "running" ? stage || "çalışıyor" : status === "done" ? "bitti" : "hata"}</span> : null}
      </div>
      {op?.output ? (
        <div className="node-port out">
          <span>{outName(step)}</span>
          <Handle type="source" position={Position.Right} id="out" style={{ top: "50%" }} />
        </div>
      ) : null}
    </div>
  );
});

const nodeTypes = { step: StepNode };

export function GraphCanvas(props: {
  steps: Step[];
  ops: Map<string, WorkflowOp>;
  models: ModelInfo[];
  job?: LiveJob;
  invalidSteps: Set<number>;
  selected: number;
  onSelect(i: number): void;
  onChange(steps: Step[]): void;
  onMessage(text: string): void;
}) {
  const { steps, ops, job } = props;

  // Status and thumbnails of the latest run, matched to nodes by id (indices change while editing).
  const runInfo = useMemo(() => {
    const info = new Map<string, { status: StepData["status"]; stage: string; thumb?: string; frames?: number }>();
    if (!job?.workflow) return info;
    job.workflow.steps.forEach((s, i) => {
      const id = stepId(s);
      if (!id) return;
      const out = job.outputs.find((o) => o.step === i + 1);
      let status: StepData["status"] = "";
      if (out) status = "done";
      else if (job.state === "running" && job.step === i + 1) status = "running";
      else if (job.state === "failed" && job.step === i + 1) status = "failed";
      info.set(id, {
        status,
        stage: job.step === i + 1 ? job.stage : "",
        thumb: out?.clip && out.frames ? frameUrl(job.id, out.clip, 0, 192) : undefined,
        frames: out?.frames,
      });
    });
    return info;
  }, [job]);

  const derived = useMemo<Node<StepData>[]>(
    () =>
      steps.map((s, i) => {
        const ui = s.ui as StepUi;
        const r = runInfo.get(ui.id);
        return {
          id: ui.id,
          type: "step",
          position: { x: ui.x, y: ui.y },
          selected: i === props.selected,
          data: { step: s, op: ops.get(s.op), index: i, status: r?.status ?? "", stage: r?.stage ?? "", thumb: r?.thumb, frames: r?.frames, invalid: props.invalidSteps.has(i) },
        };
      }),
    [steps, ops, runInfo, props.selected, props.invalidSteps],
  );
  // Local copy so dragging is smooth; positions go back to the workflow when a drag ends.
  const [nodes, setNodes] = useState(derived);
  useEffect(() => setNodes(derived), [derived]);

  // Edge selection is the only edge state React Flow owns; everything else comes from the workflow.
  const [selectedEdges, setSelectedEdges] = useState<Set<string>>(new Set());
  const edges = useMemo<Edge[]>(() => {
    const out: Edge[] = [];
    steps.forEach((s, i) => {
      for (const r of references(steps, ops, i)) {
        const src = stepId(steps[r.from]);
        const id = `${src}->${stepId(s)}:${r.field}:${r.slot}`;
        const sel = selectedEdges.has(id);
        out.push({
          id,
          source: src,
          sourceHandle: "out",
          target: stepId(s),
          targetHandle: r.field,
          label: r.field === "in" ? undefined : fieldLabel(r.field),
          animated: runInfo.get(stepId(s))?.status === "running",
          selected: sel,
          style: { stroke: sel ? "#8ab4ff" : "#6b7385", strokeWidth: sel ? 3 : 2 },
          labelStyle: { fill: "#a3a9b7", fontSize: 12 },
          labelBgStyle: { fill: "#0f1115" },
        });
      }
    });
    return out;
  }, [steps, ops, runInfo, selectedEdges]);

  function onEdgesChange(changes: EdgeChange<Edge>[]) {
    setSelectedEdges((cur) => {
      const next = new Set(cur);
      for (const c of changes) {
        if (c.type === "select") c.selected ? next.add(c.id) : next.delete(c.id);
        else if (c.type === "remove") next.delete(c.id);
      }
      return next;
    });
  }

  const indexOf = (id: string) => steps.findIndex((s) => stepId(s) === id);

  function onNodesChange(changes: NodeChange<Node<StepData>>[]) {
    setNodes((ns) => applyNodeChanges(changes, ns));
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length) props.onChange(steps.filter((s) => !removed.includes(stepId(s))));
    for (const c of changes) if (c.type === "select" && c.selected) props.onSelect(indexOf(c.id));
  }

  function onNodeDragStop(_: unknown, node: Node) {
    props.onChange(steps.map((s) => (stepId(s) === node.id ? { ...s, ui: { ...(s.ui as StepUi), x: Math.round(node.position.x), y: Math.round(node.position.y) } } : s)));
  }

  function onConnect(c: Connection) {
    const from = indexOf(c.source), to = indexOf(c.target);
    if (from < 0 || to < 0 || from === to || !c.targetHandle) return;
    let next = steps.map((s) => ({ ...s }));
    if (next[from].out === undefined) next[from].out = uniqueName(next, next[from].op);
    const name = outName(next[from]);
    const field = clipFields(ops.get(next[to].op)).find((f) => f.name === c.targetHandle);
    if (!field) return;
    if (field.type === "clips") {
      const cur = next[to][field.name];
      const list = Array.isArray(cur) ? (cur as string[]) : typeof cur === "string" ? [cur] : [];
      next[to][field.name] = [...list, name];
    } else {
      next[to][field.name] = name;
    }
    const sorted = topoSort(next, ops);
    if (!sorted) {
      props.onMessage("Bu bağlantı bir döngü oluşturur: bir adım kendi sonucunu bekleyemez.");
      return;
    }
    next = sorted;
    props.onChange(next);
  }

  function onEdgesDelete(deleted: Edge[]) {
    let next = steps.map((s) => ({ ...s }));
    for (const e of deleted) {
      const to = next.findIndex((s) => stepId(s) === e.target);
      const from = next.findIndex((s) => stepId(s) === e.source);
      if (to < 0 || from < 0 || !e.targetHandle) continue;
      const name = outName(next[from]);
      const v = next[to][e.targetHandle];
      if (Array.isArray(v)) {
        const list = (v as string[]).filter((x) => x !== name);
        if (list.length) next[to][e.targetHandle] = list;
        else delete next[to][e.targetHandle];
      } else if (v === name) {
        delete next[to][e.targetHandle];
      }
    }
    next = topoSort(next, ops) ?? next;
    props.onChange(next);
  }

  return (
    <div className="flow" aria-label="Düğüm grafiği">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onNodeDragStop={onNodeDragStop}
        onConnect={onConnect}
        onEdgesChange={onEdgesChange}
        onEdgesDelete={onEdgesDelete}
        onPaneClick={() => undefined}
        colorMode="dark"
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
        minZoom={0.2}
        deleteKeyCode={["Backspace", "Delete"]}
        proOptions={{ hideAttribution: false }}
      >
        <Background gap={24} color="#232834" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable nodeColor={() => "#2a2f3a"} maskColor="rgba(15,17,21,0.7)" />
      </ReactFlow>
    </div>
  );
}
