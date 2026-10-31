import { describe, it, expect, vi } from 'vitest';
import { z } from 'zod';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { WorkerPolicy } from '../src/worker-policy.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { Agent } from '../src/agent.js';
import { loadConfig } from '../src/config.js';
import { ModelManager } from '../src/models.js';
import { PermissionManager } from '../src/permissions.js';
import { startApiServer } from '../src/api.js';
import { startWorker } from '../src/worker.js';
import { createWebSearchTool } from '../src/tools/web.js';

const policyData = { schema_version: 1, role: 'research', deny_unlisted_tools: true, forbid_direct_tool_execute: true, allowed_tools: ['web_search'], max_tool_calls_per_job: 6, max_job_seconds: 5 };
const denied = ['bash', 'git', 'github', 'github_create_issue', 'docker', 'write_file', 'edit_file',
  'multi_edit', 'http_request', 'web_fetch', 'task', 'task_output', 'deep_research', 'update_memory',
  'todo_write', 'mcp_write_file', 'remote_execute', 'tool_search', 'unknown_future_tool'];
const token = 'test-only-'.repeat(5);

function registry() {
  const r = new ToolRegistry(new WorkerPolicy(policyData));
  for (const name of ['web_search', ...denied]) r.register({ name, description: name, schema: z.object({}), execute: vi.fn(async () => ({ success: true, output: 'executed' })) });
  return r;
}

describe('immutable worker policy boundary', () => {
  it.each([{}, { ...policyData, schema_version: 2 }, { ...policyData, allowed_tools: ['web_search', 'web_search'] }, { ...policyData, extra: true }])('rejects malformed policies', value => {
    expect(() => new WorkerPolicy(value)).toThrow();
  });
  it('fails closed for missing/unavailable policy and unknown implementation', () => {
    expect(() => WorkerPolicy.load('')).toThrow();
    expect(() => WorkerPolicy.load('/missing-policy')).toThrow();
    expect(() => new WorkerPolicy(policyData).validateAvailable([])).toThrow();
  });
  it('intersects omitted, empty and hostile client lists', () => {
    const p = new WorkerPolicy(policyData);
    expect(p.intersect()).toEqual(['web_search']);
    expect(p.intersect([])).toEqual([]);
    expect(p.intersect(['bash', 'web_search'])).toEqual(['web_search']);
  });
  it.each(denied)('denies %s before invoking registered implementation', async name => {
    const r = registry();
    expect((await r.execute(name, {})).success).toBe(false);
    expect(r.get(name)!.execute).not.toHaveBeenCalled();
  });
  it('enforces total tool-call budget across calls and resets only for a new run', async () => {
    const r = registry();
    for (let i = 0; i < 6; i++) expect((await r.execute('web_search', {})).success).toBe(true);
    expect((await r.execute('web_search', {})).error).toMatch(/budget/);
    r.beginRun();
    expect((await r.execute('web_search', {})).success).toBe(true);
  });
  it('executes approved tool, advertises only approved tools and rejects replacement', async () => {
    const r = registry();
    expect(r.names()).toEqual(['web_search']);
    expect(r.toOllamaTools().map(t => t.function.name)).toEqual(['web_search']);
    expect((await r.execute('web_search', {})).success).toBe(true);
    expect(() => r.register(r.get('web_search')!)).toThrow();
  });
  it('preserves unrestricted registries for upstream profiles', async () => {
    const r = new ToolRegistry();
    r.register({ name: 'bash', description: '', schema: z.object({}), execute: async () => ({ success: true, output: 'upstream' }) });
    expect((await r.execute('bash', {})).output).toBe('upstream');
  });
});

async function apiHarness(toolName = 'web_search') {
  let requests: any[] = [];
  let hang = toolName === "__hang__";
  const backend = createServer(async (req, res) => {
    if (req.url === '/v1/models') { res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return; }
    let body = ''; for await (const chunk of req) body += chunk;
    const data = JSON.parse(body); requests.push(data);
    if (hang) return;
    res.setHeader('Content-Type', 'text/event-stream');
    const hasResult = data.messages.some((m: any) => m.role === 'tool');
    const delta = hasResult ? { content: 'Research complete' } : { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: toolName, arguments: '{}' } }] };
    res.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: hasResult ? 'stop' : 'tool_calls' }] })}\n\ndata: [DONE]\n\n`);
  });
  backend.listen(0, '127.0.0.1'); await once(backend, 'listening');
  const address = backend.address() as any;
  const config = { ...loadConfig(''), proxyUrl: '', llmBackend: 'openai' as const, openaiBaseUrl: `http://127.0.0.1:${address.port}/v1`, model: 'test-model', lockModel: 'test-model', autoSwitch: false };
  const r = registry(); const m = new ModelManager(config); await m.discover(); m.switchTo('test-model');
  const a = new Agent(config, r, m, new PermissionManager());
  const api = startApiServer({ port: 0, agent: a, modelManager: m, registry: r, apiToken: token });
  await new Promise(resolve => setTimeout(resolve, 30));
  return { r, a, m, requests, recover: () => { hang = false; }, url: `http://127.0.0.1:${api.port}`, close: () => { api.close(); backend.close(); } };
}

describe('actual upstream Agent and HTTP API with mock inference', () => {
  it.each(denied)('rejects model-generated %s through the agent loop', async name => {
    const h = await apiHarness(name);
    try {
      const result = await h.a.runSync('test', { permissionMode: 'auto_allow' });
      expect(result.toolCalls.some(t => t.name === name && !t.success)).toBe(true);
      expect(h.r.get(name)!.execute).not.toHaveBeenCalled();
      expect(h.requests[0].tools.map((t: any) => t.function.name)).toEqual(['web_search']);
    } finally { h.close(); }
  });
  it('requires auth on all routes and disallows execute and RC even with valid auth', async () => {
    const h = await apiHarness();
    try {
      for (const path of ['/health', '/v1/models', '/api/tools', '/api/status', '/rc/stream']) {
        expect((await fetch(h.url + path)).status).toBe(401);
        expect((await fetch(h.url + path, { headers: { authorization: 'Bearer incorrect' } })).status).toBe(401);
      }
      expect((await fetch(h.url + '/api/execute', { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: '{"tool":"web_search"}' })).status).toBe(403);
      for (const bad of [{ apiExecute: true }, { rcEnabled: true }, { apiToken: '' }, { teacher: { enabled: true } }]) {
        expect(() => startApiServer({ port: 0, agent: h.a, modelManager: h.m, registry: h.r, apiToken: token, ...bad } as any)).toThrow();
      }
    } finally { h.close(); }
  });
  it.each([undefined, [], [{ function: { name: 'bash' } }], [{ function: { name: 'web_search' } }]])('client list %j cannot expand authority; requests start fresh', async tools => {
    const h = await apiHarness();
    try {
      const response = await fetch(h.url + '/v1/chat/completions', { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: JSON.stringify({ tools, messages: [{ role: 'user', content: 'test @/etc/passwd and /tmp/secret.png' }] }) });
      expect(response.status).toBe(200);
      expect(h.requests[0].messages.find((m: any) => m.role === 'user').content).toBe('test @/etc/passwd and /tmp/secret.png');
      const advertised = h.requests[0].tools || [];
      expect(advertised.every((t: any) => t.function.name === 'web_search')).toBe(true);
      if (tools && !tools.some(t => t.function.name === 'web_search')) expect(h.r.get('web_search')!.execute).not.toHaveBeenCalled();
    } finally { h.close(); }
  });
});

describe('worker boot and bounded search', () => {
  it('refuses boot without protected configuration', async () => { await expect(startWorker({})).rejects.toThrow(); });
  it('rejects unaudited tool policy at startup', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'vcode-policy-'));
    try {
      writeFileSync(join(dir, 'policy'), JSON.stringify({ ...policyData, allowed_tools: ['bash'] }));
      await expect(startWorker({ CONDUCTOR_WORKER_POLICY_PATH: join(dir, 'policy') })).rejects.toThrow(/Unenforceable/);
    } finally { rmSync(dir, { recursive: true }); }
  });
  it('rejects invalid search args, bounds results, refuses redirects and oversized bodies', async () => {
    const server = createServer((req, res) => {
      const q = new URL(req.url!, 'http://localhost').searchParams.get('q');
      if (q === 'redirect') { res.writeHead(302, { Location: 'http://127.0.0.1:1/private' }); res.end(); }
      else if (q === 'huge') res.end('x'.repeat(300000));
      else res.end(JSON.stringify({ results: Array.from({ length: 30 }, () => ({ title: 't'.repeat(1000), url: 'https://example.com', content: 'c'.repeat(10000) })) }));
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening');
    try {
      const r = new ToolRegistry(new WorkerPolicy(policyData));
      r.register(createWebSearchTool(`http://127.0.0.1:${(server.address() as any).port}`));
      expect((await r.execute('web_search', { query: 'x', max_results: 11 })).success).toBe(false);
      expect((await r.execute('web_search', { query: 'redirect' })).success).toBe(false);
      expect((await r.execute('web_search', { query: 'huge' })).success).toBe(false);
      // 30 large entries exceed the response ceiling, rather than being buffered unboundedly.
      expect((await r.execute('web_search', { query: 'normal' })).success).toBe(false);
    } finally { server.close(); }
  });
});

describe('worker startup and recovery', () => {
  it('starts authenticated, rejects bind conflicts rather than hunting ports, and can restart', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'vcode-boot-'));
    const portProbe = createServer(); portProbe.listen(0, '127.0.0.1'); await once(portProbe, 'listening');
    const port = (portProbe.address() as any).port; await new Promise<void>(resolve => portProbe.close(() => resolve()));
    const env = { CONDUCTOR_WORKER_POLICY_PATH: join(dir, 'policy'), VCODE_WORKER_SETTINGS: join(dir, 'settings'), VCODE_API_TOKEN_FILE: join(dir, 'token') };
    writeFileSync(env.CONDUCTOR_WORKER_POLICY_PATH, JSON.stringify(policyData));
    writeFileSync(env.VCODE_WORKER_SETTINGS, JSON.stringify({ openaiBaseUrl: 'http://127.0.0.1:1/v1', model: 'test-model', searxngUrl: 'http://127.0.0.1:2', apiPort: port }));
    writeFileSync(env.VCODE_API_TOKEN_FILE, token, { mode: 0o600 });
    let worker: Awaited<ReturnType<typeof startWorker>> | undefined;
    try {
      worker = await startWorker(env);
      expect((await fetch(`http://127.0.0.1:${port}/health`, { headers: { authorization: `Bearer ${token}` } })).status).toBe(200);
      await expect(startWorker(env)).rejects.toThrow(/EADDRINUSE/);
      worker.api.close(); worker = undefined;
      await new Promise(resolve => setTimeout(resolve, 30));
      worker = await startWorker(env);
      expect(worker.registry.names()).toEqual(['web_search']);
    } finally { worker?.api.close(); rmSync(dir, { recursive: true }); }
  });
  it('enforces streaming denials and recovers for a following request', async () => {
    const h = await apiHarness('bash');
    try {
      for (let i = 0; i < 2; i++) {
        const response = await fetch(h.url + '/v1/chat/completions', { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: JSON.stringify({ stream: true, messages: [{ role: 'user', content: 'test' }] }) });
        expect(response.status).toBe(200);
        expect(await response.text()).toContain('"success":false');
        expect(h.a.isRunning()).toBe(false);
      }
      expect(h.r.get('bash')!.execute).not.toHaveBeenCalled();
    } finally { h.close(); }
  });
});


describe('inference failure deadline', () => {
  it('aborts a hung inference, releases the lock, and serves a fresh subsequent job', async () => {
    const h = await apiHarness('__hang__');
    try {
      await expect(fetch(h.url + '/v1/chat/completions', { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: JSON.stringify({ messages: [{ role: 'user', content: 'test' }] }) })).rejects.toThrow();
      for (let i = 0; i < 20 && h.a.isRunning(); i++) await new Promise(resolve => setTimeout(resolve, 25));
      expect(h.a.isRunning()).toBe(false);
      h.recover();
      const response = await fetch(h.url + '/v1/chat/completions', { method: 'POST', headers: { authorization: `Bearer ${token}` }, body: JSON.stringify({ tools: [], messages: [{ role: 'user', content: 'test' }] }) });
      expect(response.status).toBe(200);
    } finally { h.close(); }
  }, 10000);
});
