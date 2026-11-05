// Engine state for the whole UI: info, schema, models and live jobs.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, subscribe, type EngineInfo, type Job, type ModelInfo, type Schema, type Workflow } from "./api";

export interface LiveJob extends Job {
  preview?: string;   // URL of the latest live preview
  workflow?: Workflow; // what was sent (jobs started in this tab)
  title?: string;
  log: string[];
}

export interface Engine {
  info?: EngineInfo;
  schema?: Schema;
  models: ModelInfo[];
  jobs: LiveJob[]; // newest first
  connected: boolean;
  loadError?: string;
  run(wf: Workflow, title: string): Promise<number>;
  cancel(id: number): Promise<void>;
  refreshModels(): Promise<void>;
}

export function useEngine(): Engine {
  const [info, setInfo] = useState<EngineInfo>();
  const [schema, setSchema] = useState<Schema>();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [jobs, setJobs] = useState<Map<number, LiveJob>>(new Map());
  const [connected, setConnected] = useState(false);
  const [loadError, setLoadError] = useState<string>();
  const meta = useRef(new Map<number, { workflow: Workflow; title: string }>());

  const merge = useCallback((id: number, patch: Partial<LiveJob>) => {
    setJobs((prev) => {
      const next = new Map(prev);
      const old = next.get(id);
      const m = meta.current.get(id);
      next.set(id, { ...(old ?? ({ id, log: [], outputs: [] } as unknown as LiveJob)), ...m, ...patch, id } as LiveJob);
      return next;
    });
  }, []);

  const load = useCallback(async () => {
    try {
      const [i, s, m, j] = await Promise.all([api.info(), api.schema(), api.models(), api.jobs()]);
      setInfo(i);
      setSchema(s);
      setModels(m);
      setJobs((prev) => {
        const next = new Map(prev);
        for (const job of j) next.set(job.id, { ...(next.get(job.id) ?? { log: [] }), ...job } as LiveJob);
        return next;
      });
      setLoadError(undefined);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void load();
    return subscribe(
      (e) => {
        switch (e.type) {
          case "job":
            merge(e.data.id, e.data);
            break;
          case "progress":
            merge(e.data.job, { step: e.data.step, steps: e.data.steps, op: e.data.op, stage: e.data.stage, fraction: e.data.fraction, state: "running" });
            break;
          case "preview":
            merge(e.data.job, { preview: e.data.url });
            break;
          case "output":
            setJobs((prev) => {
              const next = new Map(prev);
              const old = next.get(e.data.job);
              if (!old) return prev;
              const outputs = [...old.outputs.filter((o) => o.step !== e.data.step), { ...e.data, width: 0, height: 0 }];
              next.set(e.data.job, { ...old, outputs, clipsKept: true });
              return next;
            });
            break;
          case "log":
            setJobs((prev) => {
              const next = new Map(prev);
              const old = next.get(e.data.job);
              if (!old) return prev;
              next.set(e.data.job, { ...old, log: [...old.log, e.data.line].slice(-200) });
              return next;
            });
            break;
        }
      },
      (on) => {
        setConnected(on);
        if (on) void load(); // catch up after a reconnect
      },
    );
  }, [load, merge]);

  const run = useCallback(
    async (wf: Workflow, title: string) => {
      const { job } = await api.run(wf);
      meta.current.set(job, { workflow: wf, title });
      // Events may have arrived before this response (a fully cached job finishes at once): keep their state.
      setJobs((prev) => {
        const next = new Map(prev);
        const old = next.get(job);
        next.set(job, { ...(old ?? ({ id: job, log: [], outputs: [], state: "queued" } as unknown as LiveJob)), workflow: wf, title, id: job });
        return next;
      });
      return job;
    },
    [],
  );

  const cancel = useCallback(async (id: number) => {
    await api.cancel(id);
  }, []);

  const refreshModels = useCallback(async () => {
    setModels(await api.models());
  }, []);

  const list = [...jobs.values()].sort((a, b) => b.id - a.id);
  return { info, schema, models, jobs: list, connected, loadError, run, cancel, refreshModels };
}
