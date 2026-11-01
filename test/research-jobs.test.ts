import { describe, it, expect, vi } from 'vitest';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { ResearchJobs } from '../src/research-jobs.js';
import { WorkerPolicy } from '../src/worker-policy.js';
import { ToolRegistry } from '../src/tools/registry.js';
import type { Agent, AgentEvent } from '../src/agent.js';

const policy = { schema_version: 1, role: 'research', deny_unlisted_tools: true, forbid_direct_tool_execute: true, allowed_tools: ['web_search'], max_tool_calls_per_job: 6, max_job_seconds: 120 };
function registry() {
  const r = new ToolRegistry(new WorkerPolicy(policy));
  r.register({ name: 'web_search', description: '', schema: z.object({ query: z.string(), max_results: z.number().default(5) }), execute: vi.fn(async () => ({ success: true, output: 'docs', sources: ['https://docs.searxng.org/'] })) });
  return r;
}
const tick = () => new Promise(resolve => setTimeout(resolve, 10));
function harness() {
  const r = registry(); let running = false; let finish: () => void = () => {}; let telemetry: ((e: any) => void) | undefined;
  const a = { clear: vi.fn(), isRunning: () => running, setTelemetry: (f: any) => { telemetry = f; }, abort: vi.fn(() => finish()),
    run: () => { running = true; r.beginRun(); const wait = new Promise<void>(resolve => { finish = resolve; });
      return (async function*(): AsyncGenerator<AgentEvent> { try {
        telemetry?.({ kind: 'model_turn_started' });
        const result = await r.execute('web_search', { query: 'docs' });
        yield { type: 'tool_result', name: 'web_search', success: true, content: result.output, sources: result.sources };
        yield { type: 'text', content: 'See https://docs.searxng.org/' }; await wait;
        telemetry?.({ kind: 'model_turn_finished', duration_ms: 12, output_tokens: 5 });
      } finally { running = false; } })(); },
  } as unknown as Agent;
  const jobs = new ResearchJobs(a, r);
  const req = () => ({ contract_version: 1, capability: 'research.web', expected_worker_instance: jobs.instance, task_id: randomUUID(), attempt_id: randomUUID(), correlation_id: randomUUID(), idempotency_key: randomUUID(), query: 'docs', max_results: 3, deadline_seconds: 120 });
  return { r, a, jobs, req, finish: () => finish() };
}
describe('job-scoped successful search reuse', () => {
  it('deduplicates concurrent and repeated searches but preserves changed arguments and independent jobs', async () => {
    const r = registry(); const fn = r.get('web_search')!.execute;
    await Promise.all([r.execute('web_search', { query: 'x' }), r.execute('web_search', { max_results: 5, query: 'x' })]);
    await r.execute('web_search', { query: 'x' }); expect(fn).toHaveBeenCalledTimes(1);
    await r.execute('web_search', { query: 'y' }); await r.execute('web_search', { query: 'x', max_results: 3 });
    expect(fn).toHaveBeenCalledTimes(3); expect(r.diagnostics.cache_hits).toBe(2);
    r.beginRun(); await r.execute('web_search', { query: 'x' }); expect(fn).toHaveBeenCalledTimes(4);
  });
  it('does not cache failed searches', async () => {
    const r = registry(); vi.mocked(r.get('web_search')!.execute).mockResolvedValueOnce({ success: false, output: '', error: 'temporary' });
    await r.execute('web_search', { query: 'x' }); await r.execute('web_search', { query: 'x' });
    expect(r.get('web_search')!.execute).toHaveBeenCalledTimes(2);
  });
});
describe('ephemeral authenticated executor receipts', () => {
  it('reserves one job, supports idempotency, rejects identity conflicts and obsolete worker instances', async () => {
    const h = harness(); const req = h.req(); expect(h.jobs.submit(req).code).toBe(202);
    expect(h.jobs.submit(req).code).toBe(200); expect(h.jobs.submit({ ...req, query: 'changed' })).toMatchObject({ code: 409, body: { contract_version: 1, error_code: 'identity_conflict' } });
    expect(h.jobs.submit(h.req())).toMatchObject({ code: 409, body: { contract_version: 1, error_code: 'worker_busy' } }); expect(h.jobs.submit({ ...h.req(), expected_worker_instance: randomUUID() })).toMatchObject({ code: 409, body: { contract_version: 1, error_code: 'worker_instance_mismatch' } });
    expect(h.jobs.submit({ ...h.req(), deadline_seconds: 121 }).code).toBe(422);
    h.finish(); await tick(); const done = h.jobs.get(req.attempt_id)!;
    expect(done.status).toBe('completed'); expect(done.released).toBe(true); expect(done.tools[0].sources).toEqual(done.sources);
    expect(done.diagnostics.model_turns).toBe(1);
    expect((h.jobs.cancel(req.attempt_id, h.jobs.instance, 'cancelled').body as any).status).toBe('completed');
  });
  it('confirms cancellation only after underlying tool work and agent stream settle', async () => {
    const h = harness(); let resolve!: (v: any) => void;
    vi.mocked(h.r.get('web_search')!.execute).mockImplementationOnce(() => new Promise(r => { resolve = r; }));
    const req = h.req(); h.jobs.submit(req);
    const ack = h.jobs.cancel(req.attempt_id, h.jobs.instance, 'cancelled').body as any;
    expect(ack.status).toBe('cancellation_requested'); expect(ack.released).toBe(false); expect(h.jobs.status().busy).toBe(true);
    expect(h.jobs.cancel(req.attempt_id, randomUUID(), 'cancelled')).toMatchObject({ code: 409, body: { contract_version: 1, error_code: 'worker_instance_mismatch' } });
    resolve({ success: true, output: '', sources: [] }); await tick();
    expect(h.jobs.get(req.attempt_id)!.status).toBe('cancelled'); expect(h.jobs.get(req.attempt_id)!.released).toBe(true);
    expect(new ResearchJobs(h.a, h.r).get(req.attempt_id)).toBeNull();
  });
  it('enforces a shorter job deadline without changing server policy', async () => {
    const h = harness(); const req = { ...h.req(), deadline_seconds: 1 }; h.jobs.submit(req); await new Promise(r => setTimeout(r, 1050));
    expect(h.jobs.get(req.attempt_id)!.status).toBe('timed_out'); expect(h.r.policy!.deadlineMs).toBe(120000);
  });
  it('bounds event arguments and UTF8 output', async () => {
    const h = harness(); const req = h.req(); h.jobs.submit(req); h.finish(); await tick();
    expect(Buffer.byteLength(JSON.stringify(h.jobs.get(req.attempt_id)))).toBeLessThan(65536);
  });
});
