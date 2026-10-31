import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';

describe('the unauthenticated route list', () => {
  // This set is the only thing standing between a bearer token and the open
  // internet-facing surface of vcode. It exists so the login screen can load
  // its own JS; every addition to it must be an inert asset, never a route that
  // reads or changes state.
  const api = readFileSync(new URL('../src/api.ts', import.meta.url), 'utf-8');

  it('exempts only the login page, its assets, and the SSE stream', () => {
    const match = api.match(/const RC_PUBLIC = new Set\(\[([^\]]*)\]\)/);
    expect(match, 'RC_PUBLIC set not found in api.ts').toBeTruthy();
    const entries = [...match![1].matchAll(/'([^']+)'/g)].map((m) => m[1]).sort();
    expect(entries).toEqual(['/rc', '/rc/', '/rc/app.css', '/rc/app.js', '/rc/stream']);
  });

  it('never exempts anything but GET', () => {
    expect(api).toContain("const skipAuth = !registry.policy && req.method === 'GET' && RC_PUBLIC.has(reqPath);");
  });

  it('serves web assets from an allow-list, so no path can traverse', () => {
    const rc = readFileSync(new URL('../src/rc.ts', import.meta.url), 'utf-8');
    expect(rc).toContain("const WEB_ASSETS = new Set(['app.js', 'app.css']);");
  });

  it('resolves the bundle from the module, not the cwd', () => {
    // The agent chdirs into the project it is working on, so a cwd-relative
    // path would find the UI only when vcode was started from its own repo.
    const rc = readFileSync(new URL('../src/rc.ts', import.meta.url), 'utf-8');
    expect(rc).toContain('dirname(fileURLToPath(import.meta.url))');
    expect(rc).not.toMatch(/WEB_DIR\s*=\s*join\(process\.cwd\(\)/);
  });
});

describe('Remote Connect source invariants', () => {
  it('broadcasts remote user messages as a dedicated user_message event', () => {
    const source = readFileSync(new URL('../src/rc.ts', import.meta.url), 'utf-8');

    expect(source).toContain("broadcast('user_message', { content: data.message });");
    expect(source).not.toContain("broadcast('text', { role: 'user', content: data.message });");
  });

  it('exposes an explicit RC abort endpoint that calls agent.abort()', () => {
    const source = readFileSync(new URL('../src/rc.ts', import.meta.url), 'utf-8');

    expect(source).toContain("if (path === '/rc/abort' && req.method === 'POST')");
    expect(source).toContain('agentFor().abort();');
  });

  it('resets message history before replaying history after reconnect', () => {
    const source = readFileSync(new URL('../src/rc-ui.ts', import.meta.url), 'utf-8');

    expect(source).toContain('let historyResetPending = false;');
    expect(source).toContain("if (historyResetPending) {");
    expect(source).toContain("document.getElementById('messages').innerHTML = '';");
  });

  it('renders remote user_message events as user bubbles and stop uses /rc/abort', () => {
    const source = readFileSync(new URL('../src/rc-ui.ts', import.meta.url), 'utf-8');

    expect(source).toContain("eventSource.addEventListener('user_message'");
    expect(source).toContain("addMessage('user', data.content);");
    expect(source).toContain("fetch(API_BASE + '/rc/abort'");
    expect(source).not.toContain("body: JSON.stringify({ message: '/stop' })");
  });
});
