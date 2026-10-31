/** Dependency probes are separate from process liveness and never restart it. */
export interface WorkerDependencies {
  openaiBaseUrl: string;
  model: string;
  searxngUrl: string;
}
export interface WorkerReadiness {
  ready: boolean;
  dependencies: { dgxModel: boolean; searxngSearch: boolean };
}
const MAX_PROBE_BYTES = 256 * 1024;

async function boundedJson(url: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(url, { signal, redirect: 'error', headers: { Accept: 'application/json' } });
  if (!response.ok || !response.body) throw new Error('Dependency unavailable');
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_PROBE_BYTES) throw new Error('Dependency response exceeds limit');
      chunks.push(value);
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } finally { await reader.cancel().catch(() => undefined); }
}

/** Fixed operator endpoints, no client parameters, retries, model loads or generation.
 * A single shared deadline covers headers AND bodies; concurrent callers share a probe.
 * Readiness proves the configured alias is advertised and search JSON is usable,
 * not that generation or every upstream search engine will succeed. */
export function createWorkerReadiness(settings: WorkerDependencies, timeoutMs = 3000): () => Promise<WorkerReadiness> {
  let pending: Promise<WorkerReadiness> | undefined;
  return () => {
    if (pending) return pending;
    const signal = AbortSignal.timeout(timeoutMs);
    const base = settings.openaiBaseUrl.replace(/\/+$/, '').replace(/\/v1$/, '');
    const model = boundedJson(`${base}/v1/models`, signal).then(value => {
      const data = value as { data?: Array<{ id?: string }> } | null;
      return Array.isArray(data?.data) && data.data.some(m => m?.id === settings.model);
    }).catch(() => false);
    const search = boundedJson(`${settings.searxngUrl.replace(/\/+$/, '')}/search?q=SearXNG&format=json&language=en`, signal).then(value => {
      const data = value as { results?: unknown } | null;
      return Array.isArray(data?.results);
    }).catch(() => false);
    pending = Promise.all([model, search]).then(([dgxModel, searxngSearch]) => ({
      ready: dgxModel && searxngSearch, dependencies: { dgxModel, searxngSearch },
    })).finally(() => { pending = undefined; });
    return pending;
  };
}
