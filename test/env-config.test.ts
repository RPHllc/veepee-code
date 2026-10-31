/**
 * ~/.veepee-code/.env holds endpoints and secrets; settings.json holds the
 * rest. Each setting lives in exactly one of them.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync, statSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { parseEnv, updateEnvFile } from '../src/env-file.js';
import { loadConfig, migrateToEnvFile, updateGlobalConfig, readGlobalConfig, answers } from '../src/config.js';
import { createServer } from 'net';

let home: string;
let cwd: string;
const saved = { HOME: process.env.HOME, cwd: process.cwd() };
const ENV_NAMES = ['VEEPEE_CODE_PROXY_URL', 'VEEPEE_CODE_LLM_BACKEND', 'VEEPEE_CODE_OPENAI_BASE_URL', 'SEARXNG_URL', 'AGENTLENS_URL', 'VEEPEE_CODE_API_TOKEN'];
const savedEnv: Record<string, string | undefined> = {};

const dir = () => join(home, '.veepee-code');
const envText = () => readFileSync(join(dir(), '.env'), 'utf-8');
const settings = () => JSON.parse(readFileSync(join(dir(), 'settings.json'), 'utf-8'));
const writeSettings = (o: unknown) => writeFileSync(join(dir(), 'settings.json'), JSON.stringify(o));

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'vcode-env-'));
  cwd = mkdtempSync(join(tmpdir(), 'vcode-cwd-'));
  mkdirSync(dir(), { recursive: true });
  process.env.HOME = home;
  process.chdir(cwd); // no project .veepee/ layer
  for (const n of ENV_NAMES) { savedEnv[n] = process.env[n]; delete process.env[n]; }
});
afterEach(() => {
  process.env.HOME = saved.HOME;
  process.chdir(saved.cwd);
  for (const n of ENV_NAMES) { if (savedEnv[n] === undefined) delete process.env[n]; else process.env[n] = savedEnv[n]; }
  rmSync(home, { recursive: true, force: true });
  rmSync(cwd, { recursive: true, force: true });
});

describe('parseEnv / updateEnvFile', () => {
  it('parses quotes, export, comments, and keeps empty values', () => {
    const m = parseEnv('# c\nexport A=1\nB="two words"\nC=\nD=x # note\n');
    expect([...m]).toEqual([['A', '1'], ['B', 'two words'], ['C', ''], ['D', 'x']]);
  });

  it('edits in place, keeping comments and unknown keys, and writes 0600', () => {
    const p = join(dir(), '.env');
    writeFileSync(p, '# mine\nFOO=bar\nSEARXNG_URL=old\nSEARXNG_URL=dup\n');
    updateEnvFile(p, { SEARXNG_URL: 'new', VEEPEE_CODE_PROXY_URL: '', FOO: null });
    expect(readFileSync(p, 'utf-8')).toBe('# mine\nSEARXNG_URL=new\nVEEPEE_CODE_PROXY_URL=\n');
    expect(statSync(p).mode & 0o777).toBe(0o600);
  });
});

describe('migrateToEnvFile', () => {
  it('moves endpoints and secrets out of an existing settings.json', () => {
    writeSettings({
      proxyUrl: 'https://gw', llmBackend: 'openai', openaiBaseUrl: 'http://gpu:8000', apiToken: 't0k',
      searxngUrl: 'http://s:8888', lockModel: 'm', fleet: [{ name: 'a', url: 'u' }],
      remote: { url: 'http://r', apiKey: 'k', allow: ['x'] },
    });
    expect(migrateToEnvFile()).toBe(true);

    const env = parseEnv(envText());
    expect(env.get('VEEPEE_CODE_PROXY_URL')).toBe('https://gw');
    expect(env.get('VEEPEE_CODE_OPENAI_BASE_URL')).toBe('http://gpu:8000');
    expect(env.get('VEEPEE_CODE_API_TOKEN')).toBe('t0k');
    expect(env.get('VEEPEE_CODE_REMOTE_API_KEY')).toBe('k');
    // Nothing in two places: settings.json keeps only the structured rest.
    expect(settings()).toEqual({ lockModel: 'm', fleet: [{ name: 'a', url: 'u' }], remote: { allow: ['x'] } });
    expect(readdirSync(dir()).some(f => f.startsWith('settings.json.bak-envsplit-'))).toBe(true);

    // And the loaded config is unchanged by the move.
    const c = loadConfig();
    expect(c.proxyUrl).toBe('https://gw');
    expect(c.apiToken).toBe('t0k');
    expect(c.remote).toEqual({ allow: ['x'], url: 'http://r', apiKey: 'k' });
    expect(c.lockModel).toBe('m');
  });

  it('writes a former default only if it is exactly the old address', () => {
    writeSettings({ lockModel: 'm' });
    migrateToEnvFile();
    // On offline/CI hosts migration may correctly create no .env at all.
    const env = existsSync(join(dir(), '.env')) ? parseEnv(envText()) : new Map<string, string>();
    if (env.has('SEARXNG_URL')) expect(env.get('SEARXNG_URL')).toBe('http://10.0.153.99:8888');
    if (env.has('AGENTLENS_URL')) expect(env.get('AGENTLENS_URL')).toBe('http://10.0.153.99:7001');
  });

  it('is a no-op the second time', () => {
    writeSettings({ proxyUrl: 'https://gw' });
    migrateToEnvFile();
    const before = envText();
    expect(migrateToEnvFile()).toBe(false);
    expect(envText()).toBe(before);
  });

  it('keeps a value already in .env over the one in settings.json', () => {
    writeFileSync(join(dir(), '.env'), 'VEEPEE_CODE_PROXY_URL=https://from-env\n');
    writeSettings({ proxyUrl: 'https://from-settings' });
    migrateToEnvFile();
    expect(loadConfig().proxyUrl).toBe('https://from-env');
    expect(settings()).toEqual({});
  });

  it('moves non-endpoint settings out of an old-style .env, and keeps the .env', () => {
    writeFileSync(join(dir(), '.env'), 'VEEPEE_CODE_PROXY_URL=http://o:11434\nVEEPEE_CODE_AUTO_SWITCH=false\nVEEPEE_CODE_API_PORT=9000\n');
    migrateToEnvFile();
    expect(envText()).toBe('VEEPEE_CODE_PROXY_URL=http://o:11434\n');
    expect(settings()).toMatchObject({ autoSwitch: false, apiPort: 9000 });
  });
});

describe('precedence', () => {
  it('settings.json < .env < project/local < process env', () => {
    writeSettings({ lockModel: 'm' });
    writeFileSync(join(dir(), '.env'), 'VEEPEE_CODE_PROXY_URL=https://env\nSEARXNG_URL=http://env-s\n');
    expect(loadConfig().proxyUrl).toBe('https://env');

    mkdirSync(join(cwd, '.veepee'));
    writeFileSync(join(cwd, '.veepee', 'settings.local.json'), JSON.stringify({ proxyUrl: 'https://local' }));
    expect(loadConfig().proxyUrl).toBe('https://local');

    process.env.VEEPEE_CODE_PROXY_URL = 'https://process';
    expect(loadConfig().proxyUrl).toBe('https://process');
    expect(loadConfig().searxngUrl).toBe('http://env-s');
  });

  it('an empty VEEPEE_CODE_PROXY_URL means no gateway', () => {
    writeSettings({});
    writeFileSync(join(dir(), '.env'), 'VEEPEE_CODE_LLM_BACKEND=openai\nVEEPEE_CODE_OPENAI_BASE_URL=http://gpu:8000\nVEEPEE_CODE_PROXY_URL=\n');
    const c = loadConfig();
    expect(c.proxyUrl).toBe('');
    expect(c.llmBackend).toBe('openai');
  });
});

describe('updateGlobalConfig', () => {
  it('routes each key to its one file and touches nothing else', () => {
    writeSettings({ fleet: [{ name: 'a', url: 'u' }], mcpServers: { x: { command: 'true' } } });
    writeFileSync(join(dir(), '.env'), '# keep me\nSEARXNG_URL=http://s\n');
    updateGlobalConfig({ model: 'qwen', autoSwitch: false, apiToken: 'secret', proxyUrl: 'https://gw' });

    expect(settings()).toEqual({ fleet: [{ name: 'a', url: 'u' }], mcpServers: { x: { command: 'true' } }, model: 'qwen', autoSwitch: false });
    expect(envText()).toBe('# keep me\nSEARXNG_URL=http://s\nVEEPEE_CODE_PROXY_URL=https://gw\nVEEPEE_CODE_API_TOKEN=secret\n');
  });

  it('does not copy a project override into the global config', () => {
    writeSettings({});
    mkdirSync(join(cwd, '.veepee'));
    writeFileSync(join(cwd, '.veepee', 'settings.local.json'), JSON.stringify({ proxyUrl: 'https://project-only' }));
    updateGlobalConfig({ progressBar: false });
    expect(readGlobalConfig().proxyUrl).toBeUndefined();
    expect(settings()).toEqual({ progressBar: false });
  });

  it('clearing remote removes its secrets and its settings', () => {
    writeSettings({ remote: { allow: ['x'] } });
    writeFileSync(join(dir(), '.env'), 'VEEPEE_CODE_REMOTE_URL=http://r\nVEEPEE_CODE_REMOTE_API_KEY=k\n');
    updateGlobalConfig({ remote: null });
    expect(parseEnv(envText()).size).toBe(0);
    expect(settings()).toEqual({});
  });

  it('refuses to overwrite an unreadable settings.json', () => {
    writeFileSync(join(dir(), 'settings.json'), '{ broken');
    expect(() => updateGlobalConfig({ model: 'x' })).toThrow(/not valid JSON/);
    expect(readFileSync(join(dir(), 'settings.json'), 'utf-8')).toBe('{ broken');
  });
});

describe('answers (former-default reachability)', () => {
  it('is true for a listening port and false for a closed one', async () => {
    const server = createServer().listen(0, '127.0.0.1');
    await new Promise((r) => server.once('listening', r));
    const port = (server.address() as { port: number }).port;
    expect(answers(`http://127.0.0.1:${port}`)).toBe(true);
    await new Promise((r) => server.close(r));
    expect(answers(`http://127.0.0.1:${port}`)).toBe(false);
    expect(answers('not a url')).toBe(false);
  });
});

describe('migrateToEnvFile is a real one-time step', () => {
  it('does nothing — no backup, no notice — when a cleared block is all there is', async () => {
    const { mkdtempSync, writeFileSync, readdirSync } = await import('fs');
    const { join } = await import('path');
    const { tmpdir } = await import('os');
    const home = mkdtempSync(join(tmpdir(), 'vcode-envsplit-'));
    const prev = process.env.HOME;
    process.env.HOME = home;
    try {
      const dir = join(home, '.veepee-code');
      (await import('fs')).mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, '.env'), 'VEEPEE_CODE_PROXY_URL=http://gw:11434\n');
      // `"sync": null` (sync switched off) re-triggered the migration on every launch.
      writeFileSync(join(dir, 'settings.json'), JSON.stringify({ sync: null, model: 'm' }, null, 2) + '\n');
      expect(migrateToEnvFile()).toBe(false);
      expect(migrateToEnvFile()).toBe(false);
      expect(readdirSync(dir).filter(f => f.includes('bak-envsplit'))).toEqual([]);
    } finally {
      process.env.HOME = prev;
    }
  });
});
