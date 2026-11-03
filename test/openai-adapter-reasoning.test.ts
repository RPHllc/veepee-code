import { describe, it, expect, afterEach } from 'vitest';
import { OpenAIChatClient } from '../src/openai-adapter.js';

/**
 * Regression: the reasoning field name is server-specific, and reading the
 * wrong one fails SILENTLY — no error, no warning, just an empty answer.
 *
 * vLLM 0.23.1 on the DGX Spark (`--reasoning-parser deepseek_r1`) emits
 * `delta.reasoning`. The adapter originally read only `delta.reasoning_content`,
 * which that server never sends. Measured against the live engine 2026-08-07:
 * with `enable_thinking: true` — vcode's default, since `think` defaults to
 * `params.think !== false` — a plain prompt returned content = 0 chars and
 * reasoning = 849 chars. A real `vcode -p "find and fix a bug"` run against it
 * read nine files over four turns and printed ONE BYTE.
 *
 * Tool calls were never affected (they arrive in `delta.tool_calls`), which is
 * why the agent appeared to be doing nothing while it was in fact working —
 * the single most misleading part of the failure.
 */

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

function sseResponse(frames: unknown[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      const enc = new TextEncoder();
      for (const f of frames) c.enqueue(enc.encode(`data: ${JSON.stringify(f)}\n\n`));
      c.enqueue(enc.encode('data: [DONE]\n\n'));
      c.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

function stubFetch(frames: unknown[]) {
  globalThis.fetch = (async () => sseResponse(frames)) as typeof fetch;
}

async function collect(client: OpenAIChatClient): Promise<{ content: string; thinking: string }> {
  const stream = await client.chat({ model: 'm', messages: [], stream: true });
  let content = '';
  let thinking = '';
  for await (const chunk of stream) {
    content += chunk.message?.content ?? '';
    thinking += (chunk.message as { thinking?: string })?.thinking ?? '';
  }
  return { content, thinking };
}

describe('OpenAIChatClient reasoning extraction', () => {
  const client = new OpenAIChatClient('http://example.invalid:8000');

  it('reads vLLM 0.23.1 `delta.reasoning`', async () => {
    stubFetch([
      { choices: [{ delta: { role: 'assistant' } }] },
      { choices: [{ delta: { reasoning: 'the sky ' } }] },
      { choices: [{ delta: { reasoning: 'is blue' } }] },
    ]);
    expect(await collect(client)).toEqual({ content: '', thinking: 'the sky is blue' });
  });

  it('still reads `delta.reasoning_content` for servers that use that name', async () => {
    stubFetch([
      { choices: [{ delta: { reasoning_content: 'alpha' } }] },
      { choices: [{ delta: { reasoning_content: 'beta' } }] },
    ]);
    expect((await collect(client)).thinking).toBe('alphabeta');
  });

  it('does not double-emit when a server sends both spellings', async () => {
    stubFetch([
      { choices: [{ delta: { reasoning: 'x', reasoning_content: 'x' } }] },
    ]);
    expect((await collect(client)).thinking).toBe('x');
  });

  /**
   * The channels must stay SEPARATE. Folding reasoning into content printed the
   * whole chain of thought above every answer and — because the agent's
   * act/verify heuristics read the assistant's own words — turned the "Let me…"
   * of ordinary reasoning into a stall signal: a greeting earned two force-act
   * nudges and two pointless bash calls (2026-08-23).
   */
  it('keeps reasoning out of content when the model answers', async () => {
    stubFetch([
      { choices: [{ delta: { reasoning: 'Let me check whether I should act. ' } }] },
      { choices: [{ delta: { content: 'Ready. What are we working on?' } }] },
    ]);
    expect(await collect(client)).toEqual({
      content: 'Ready. What are we working on?',
      thinking: 'Let me check whether I should act. ',
    });
  });

  it('emits tool calls regardless of which reasoning field is used', async () => {
    stubFetch([
      { choices: [{ delta: { reasoning: 'deciding' } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'read_file', arguments: '{"path":' } }] } }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"a.ts"}' } }] } }] },
    ]);
    const stream = await client.chat({ model: 'm', messages: [], stream: true });
    const calls: Array<{ function: { name: string; arguments: Record<string, unknown> } }> = [];
    for await (const chunk of stream) {
      if (chunk.message?.tool_calls) calls.push(...chunk.message.tool_calls);
    }
    expect(calls).toEqual([{ function: { name: 'read_file', arguments: { path: 'a.ts' } } }]);
  });
});
