import { describe, it, expect } from 'vitest';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Agent, type AgentEvent } from '../src/agent.js';
import { codingTools, CODING_TOOLS } from '../src/coding/tools.js';
import { loadConfig } from '../src/config.js';
import { ModelManager } from '../src/models.js';
import { PermissionManager } from '../src/permissions.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { WorkerPolicy } from '../src/worker-policy.js';

/** Real Agent + isolated workspace tools; only model inference is synthetic. */
async function runTools(role: string, toolTurns: number, successful = true, repeat = false) {
  const root = mkdtempSync(join(tmpdir(), 'coding-progress-'));
  for (let i = 0; i < 24; i++) writeFileSync(join(root, `file${i}.txt`), `Observed file ${i}`);
  let requests = 0;
  const backend = createServer(async (req, res) => {
    if (req.url === '/v1/models') {
      res.end(JSON.stringify({ data: [{ id: 'test-model' }] })); return;
    }
    for await (const _chunk of req) { /* Drain the bounded synthetic model request. */ }
    const turn = requests++;
    const done = turn >= toolTurns;
    const path = `${successful ? 'file' : 'missing'}${repeat ? 0 : turn}.txt`;
    const delta = done ? { content: 'Inspection complete' } : {
      tool_calls: [{ index: 0, id: `call_${turn}`, type: 'function', function: {
        name: 'read_file', arguments: JSON.stringify({ path }),
      } }],
    };
    res.setHeader('Content-Type', 'text/event-stream');
    res.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: done ? 'stop' : 'tool_calls' }] })}\n\ndata: [DONE]\n\n`);
  });
  backend.listen(0, '127.0.0.1'); await once(backend, 'listening');
  try {
    const policy = new WorkerPolicy({ schema_version: 1, role, deny_unlisted_tools: true,
      forbid_direct_tool_execute: true, allowed_tools: CODING_TOOLS,
      max_tool_calls_per_job: 64, max_job_seconds: role === 'coding' ? 600 : 180 });
    const registry = new ToolRegistry(policy);
    for (const tool of codingTools(root, ['python_unittest'], async () => ({ success: true, output: 'exit=0' }))) registry.register(tool);
    const config = { ...loadConfig(''), proxyUrl: '', llmBackend: 'openai' as const,
      openaiBaseUrl: `http://127.0.0.1:${(backend.address() as { port: number }).port}/v1`,
      model: 'test-model', lockModel: 'test-model', autoSwitch: false };
    const models = new ModelManager(config); await models.discover(); models.switchTo('test-model');
    const agent = new Agent(config, registry, models, new PermissionManager(), 'Inspect the isolated workspace using supplied tools.');
    const events: AgentEvent[] = [];
    for await (const event of agent.run('Inspect the requested files.', { permissionMode: 'auto_allow' })) events.push(event);
    return { events, requests, diagnostics: { ...registry.diagnostics }, policy };
  } finally {
    backend.closeAllConnections(); await new Promise<void>(resolve => backend.close(() => resolve()));
    rmSync(root, { recursive: true, force: true });
  }
}

describe('coding tool-only progress within unchanged worker ceilings', () => {
  it('finishes after sixteen successful distinct tool-only turns and a final answer', async () => {
    const result = await runTools('coding', 16);
    expect(result.requests).toBe(17);
    expect(result.diagnostics.actual_tool_calls).toBe(16);
    expect(result.events.filter(e => e.type === 'tool_result').every(e => e.success)).toBe(true);
    expect(result.events.some(e => e.type === 'done')).toBe(true);
    expect(result.events.filter(e => e.type === 'error')).toEqual([]);
    expect(result.policy.role).toBe('coding');
    expect(Object.isFrozen(result.policy)).toBe(true);
    expect([result.policy.maxTurns, result.policy.maxToolCalls, result.policy.deadlineMs]).toEqual([20, 64, 600000]);
  });
  it('does not count failed coding tools as progress', async () => {
    const result = await runTools('coding', 16, false);
    expect(result.requests).toBe(15);
    expect(result.events.some(e => e.type === 'error' && e.error?.includes('15 turns with no visible output'))).toBe(true);
    expect(result.events.some(e => e.type === 'done')).toBe(false);
  });
  it.each(['research', 'default'])('preserves the no-visible-output guard for %s', async role => {
    const result = await runTools(role, 16);
    expect(result.requests).toBe(15);
    expect(result.events.some(e => e.type === 'error' && e.error?.includes('15 turns with no visible output'))).toBe(true);
    expect(result.events.some(e => e.type === 'done')).toBe(false);
  });
  it('still stops coding at twenty total turns with productive tools', async () => {
    const result = await runTools('coding', 21);
    expect(result.requests).toBe(20);
    expect(result.diagnostics.actual_tool_calls).toBe(20);
    expect(result.events.some(e => e.type === 'error' && e.error?.includes('20 turns on one message'))).toBe(true);
  });
  it('still stops repeated identical successful coding calls', async () => {
    const result = await runTools('coding', 16, true, true);
    expect(result.requests).toBeLessThan(15);
    expect(result.events.some(e => e.type === 'error' && e.error?.includes('same tool call+result repeated'))).toBe(true);
    expect(result.events.some(e => e.type === 'done')).toBe(false);
  });
});
