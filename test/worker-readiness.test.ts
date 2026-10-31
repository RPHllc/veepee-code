import { describe, it, expect, vi } from 'vitest';
import { createServer, request, type Server } from 'node:http';
import { once } from 'node:events';
import { createWorkerReadiness } from '../src/worker-readiness.js';
import { startApiServer } from '../src/api.js';
import { readBody, WORKER_BODY_TIMEOUT_MS } from '../src/api-body.js';
import { WorkerPolicy } from '../src/worker-policy.js';
import { ToolRegistry } from '../src/tools/registry.js';

const token = 'test-only-'.repeat(5);
const policy = () => new WorkerPolicy({ schema_version: 1, role: 'research', deny_unlisted_tools: true,
  forbid_direct_tool_execute: true, allowed_tools: ['web_search'], max_tool_calls_per_job: 6, max_job_seconds: 1 });
async function listening(server: Server) {
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  return `http://127.0.0.1:${(server.address() as any).port}`;
}
async function dependencies() {
  const state = { model: 'valid', search: 'valid', requests: [] as string[] };
  const server = createServer((req, res) => {
    state.requests.push(req.url!);
    const model = req.url === '/v1/models';
    const mode = model ? state.model : state.search;
    if (mode === 'headers') return;
    if (mode === 'body') { res.writeHead(200); res.write('{'); return; }
    if (mode === 'redirect') { res.writeHead(302, { Location: '/v1/models' }); res.end(); return; }
    if (mode === 'error') { res.writeHead(503); res.end(); return; }
    res.end(mode === 'invalid' ? '{}' : mode === 'huge' ? 'x'.repeat(300000) :
      JSON.stringify(model ? { data: [{ id: mode === 'missing' ? 'other-model' : 'configured-model' }] } : { results: [] }));
  });
  const url = await listening(server);
  return { state, server, probe: createWorkerReadiness({ openaiBaseUrl: url + '/v1', model: 'configured-model', searxngUrl: url }, 100),
    close: () => { server.closeAllConnections(); server.close(); } };
}

describe('bounded actual-endpoint dependency probes using HTTP fixtures', () => {
  it('checks the configured alias and real search JSON, sharing concurrent probes', async () => {
    const d = await dependencies();
    try {
      const [a, b] = await Promise.all([d.probe(), d.probe()]);
      expect(a.ready).toBe(true); expect(b).toEqual(a);
      expect(d.state.requests).toHaveLength(2);
      expect(d.state.requests).toContain('/v1/models');
      expect(d.state.requests.some(p => p.startsWith('/search?') && p.includes('format=json'))).toBe(true);
    } finally { d.close(); }
  });
  it.each(['missing', 'error', 'invalid', 'huge', 'headers', 'body', 'redirect'])('reports DGX %s without trusting the locked profile', async mode => {
    const d = await dependencies();
    try {
      d.state.model = mode;
      const started = Date.now(); const result = await d.probe();
      expect(result).toEqual({ ready: false, dependencies: { dgxModel: false, searxngSearch: true } });
      expect(Date.now() - started).toBeLessThan(1000);
    } finally { d.close(); }
  });
  it.each(['error', 'invalid', 'huge', 'headers', 'body', 'redirect'])('reports search %s and recovers on the next probe', async mode => {
    const d = await dependencies();
    try {
      d.state.search = mode;
      expect((await d.probe()).dependencies).toEqual({ dgxModel: true, searxngSearch: false });
      d.state.search = 'valid'; expect((await d.probe()).ready).toBe(true);
    } finally { d.close(); }
  });
  it('keeps liveness available and auth enforced during dependency failure and recovery', async () => {
    const d = await dependencies();
    const api = startApiServer({ port: 0, registry: new ToolRegistry(policy()), apiToken: token,
      agent: {} as any, modelManager: {} as any, workerReadiness: d.probe });
    await api.ready;
    const base = `http://127.0.0.1:${api.port}`;
    const headers = { authorization: `Bearer ${token}` };
    try {
      expect((await fetch(base + '/ready')).status).toBe(401);
      expect(d.state.requests).toHaveLength(0);
      d.state.model = 'missing'; d.state.search = 'error';
      expect((await fetch(base + '/ready', { headers })).status).toBe(503);
      expect((await fetch(base + '/health', { headers })).status).toBe(200);
      d.state.model = d.state.search = 'valid';
      expect((await fetch(base + '/ready', { headers })).status).toBe(200);
      expect((await fetch(base + '/health', { headers })).status).toBe(200);
    } finally { api.close(); d.close(); }
  });
});

describe('bounded body intake', () => {
  it('clears intake timers on successful body completion', async () => {
    const read = vi.fn();
    const server = createServer(async (req, res) => {
      try { read(await readBody(req, 80)); res.end('ok'); } catch { res.end('timeout'); }
    });
    const url = await listening(server);
    try {
      expect(await (await fetch(url, { method: 'POST', body: 'complete' })).text()).toBe('ok');
      await new Promise(resolve => setTimeout(resolve, 120));
      expect(read).toHaveBeenCalledExactlyOnceWith('complete');
    } finally { server.close(); }
  });
  it('rejects aborted intake and preserves optional upstream timing', async () => {
    const done: Promise<unknown>[] = [];
    const server = createServer((req, res) => { done.push(readBody(req, req.url === '/abort' ? 80 : undefined).then(b => { res.end(b); return b; }).catch(e => e)); });
    const url = await listening(server);
    try {
      const req = request(url + '/abort', { method: 'POST' }); req.on('error', () => {}); req.write('{');
      await new Promise(resolve => setTimeout(resolve, 20)); req.destroy();
      await new Promise(resolve => setTimeout(resolve, 20));
      expect(await done[0]).toBeInstanceOf(Error);
      const result = new Promise<string>(resolve => {
        const slow = request(url, { method: 'POST' }, res => { let s = ''; res.on('data', c => s += c); res.on('end', () => resolve(s)); });
        slow.write('first'); setTimeout(() => slow.end('last'), 150);
      });
      expect(await result).toBe('firstlast');
    } finally { server.closeAllConnections(); server.close(); }
  });
  it('bounds continuous slow uploads before execution, independently of the one-second job deadline', async () => {
    const runSync = vi.fn(async () => ({ content: 'ok', toolCalls: [], errors: [], stuck: false }));
    const api = startApiServer({ port: 0, registry: new ToolRegistry(policy()), apiToken: token,
      agent: { runSync, isRunning: () => false, clear: () => {}, abort: () => {} } as any,
      modelManager: { getCurrentModel: () => 'configured-model' } as any });
    await api.ready;
    const url = `http://127.0.0.1:${api.port}/v1/chat/completions`;
    const started = Date.now();
    try {
      const status = await new Promise<number>((resolve, reject) => {
        const req = request(url, { method: 'POST', headers: { authorization: `Bearer ${token}` } }, res => {
          res.resume(); res.on('end', () => { clearInterval(drip); req.destroy(); resolve(res.statusCode!); });
        });
        const drip = setInterval(() => req.write(' '), 25);
        req.on('error', error => { clearInterval(drip); reject(error); });
        req.write('{');
      });
      expect(status).toBe(408);
      expect(Date.now() - started).toBeGreaterThanOrEqual(WORKER_BODY_TIMEOUT_MS - 100);
      expect(Date.now() - started).toBeLessThan(WORKER_BODY_TIMEOUT_MS + 3000);
      expect(runSync).not.toHaveBeenCalled();
      expect((await fetch(url, { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: JSON.stringify({ messages: [{ role: 'user', content: 'complete' }] }) })).status).toBe(200);
      expect(runSync).toHaveBeenCalledTimes(1);
    } finally { api.close(); }
  }, 15000);
});
