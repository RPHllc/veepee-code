import { afterEach, describe, expect, it, vi } from 'vitest';
import { request } from 'node:http';
import { once } from 'node:events';
import type { Server } from 'node:http';

const state = vi.hoisted(() => ({ config: {} as Record<string, unknown> }));
vi.mock('node:fs', async original => {
  const fs = await original<typeof import('node:fs')>();
  return { ...fs, readFileSync: (...args: Parameters<typeof fs.readFileSync>) =>
    args[0] === '/config/proxy.json' ? JSON.stringify(state.config) : fs.readFileSync(...args) };
});

let server: Server | undefined;
afterEach(async () => {
  if (server) await new Promise<void>(resolve => server!.close(() => resolve()));
  server = undefined;
  vi.unstubAllGlobals();
  vi.resetModules();
});

async function start(limit?: unknown) {
  state.config = { base_url: 'http://127.0.0.1:9999/v1', model: 'fixed',
    ...(limit === undefined ? {} : { max_output_tokens: limit }) };
  const upstream = vi.fn(async (..._args: [string, RequestInit]) => new Response('{"ok":true}', { headers: { 'Content-Type': 'application/json' } }));
  vi.stubGlobal('fetch', upstream);
  server = (await import('../src/coding/inference-proxy.js')).server;
  if (!server.listening) await once(server, 'listening');
  return upstream;
}

async function call(body: Record<string, unknown>, path = '/v1/chat/completions') {
  return await new Promise<number>((resolve, reject) => {
    const r = request({ host: '127.0.0.1', port: 3030, method: 'POST', path,
      headers: { 'Content-Type': 'application/json' } }, response => {
      response.resume(); response.on('end', () => resolve(response.statusCode!));
    });
    r.on('error', reject); r.end(JSON.stringify(body));
  });
}

describe('protected coding inference output budget', () => {
  it('keeps the original profile ceiling and fixed model/origin', async () => {
    const upstream = await start();
    expect(await call({ model: 'fixed', messages: [], max_tokens: 32768 })).toBe(200);
    expect(upstream.mock.calls[0][0]).toBe('http://127.0.0.1:9999/v1/chat/completions');
    expect(JSON.parse(String(upstream.mock.calls[0][1].body)).max_tokens).toBe(2048);
  });

  it('allows an enrolled implementation response but clamps model-selected expansion', async () => {
    const upstream = await start(8192);
    expect(await call({ model: 'fixed', messages: [], max_tokens: 32768 })).toBe(200);
    expect(JSON.parse(String(upstream.mock.calls[0][1].body)).max_tokens).toBe(8192);
    expect(await call({ model: 'fixed', messages: [], max_tokens: 128 })).toBe(200);
    expect(JSON.parse(String(upstream.mock.calls[1][1].body)).max_tokens).toBe(128);
    expect(await call({ model: 'other', messages: [] })).toBe(502);
    expect(await call({ model: 'fixed', messages: [] }, '/arbitrary')).toBe(403);
    expect(upstream).toHaveBeenCalledTimes(2);
  });

  it.each([null, true, 0, 255, 8193, '8192'])('rejects invalid protected ceiling %j before listening', async limit => {
    await expect(start(limit)).rejects.toThrow('output budget');
  });
});
