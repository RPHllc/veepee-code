import { resolve, join } from 'path';
import { spawnSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync } from 'fs';
import type { LspServerConfig } from './lsp/config.js';
import { writeFileAtomicSync } from './atomic-write.js';
import { readEnvFile, updateEnvFile } from './env-file.js';

export interface Config {
  proxyUrl: string;
  /** LLM backend transport. "ollama" (default) = Ollama-format `/api/chat`
   *  via `proxyUrl` (the llm-gateway). "openai" = talk DIRECTLY to a
   *  vLLM/OpenAI-compatible server's `/v1/chat/completions` at `openaiBaseUrl`,
   *  bypassing the gateway. Opt-in; default preserves current behavior. */
  llmBackend: 'ollama' | 'openai';
  /** Base URL for the openai backend (e.g. "http://10.0.154.246:8000" or
   *  ".../v1"). Only used when llmBackend === "openai". */
  openaiBaseUrl: string | null;
  /** Isolated worker generation setting; never changes tool permissions. */
  workerThinking?: boolean;
  /** Bearer token for the openai backend, if it requires one (vLLM usually
   *  does not). Only used when llmBackend === "openai". */
  openaiApiKey: string | null;
  dashboardUrl: string;
  model: string | null;
  lockModel: string | null;
  reviewModel: string | null;
  /** The model `/act 2` runs on (Shift+Tab's second stop). Falls back to the
   *  first of fallbackModels. Read from `planModel` too, its old name from
   *  when this was plan mode's model. */
  secondModel: string | null;
  /** Model used for compaction summaries. Falls back to the current chat
   *  model when null. Pin a smaller/cheaper model here to keep summaries
   *  fast and stop them from blocking the main loop on a large model. */
  summarizerModel: string | null;
  autoSwitch: boolean;
  /** Context window to request from Ollama (num_ctx). null = the model's own
   *  maximum, capped at 16384 (see ensureContextSize in agent.ts). */
  numCtx: number | null;
  /** Models to continue on, in order, when the current one cannot be reached
   *  or drops the connection mid-reply. Empty = no fallback. */
  fallbackModels: string[];
  /** When connected MCP servers add more tools than this, they are loaded on
   *  demand via tool_search instead of sent with every request. */
  mcpDeferAbove: number;
  maxModelSize: number;  // max parameter count in billions (default 40)
  minModelSize: number;  // min for act mode — skip tiny models (default 12)
  apiPort: number;
  apiHost: string;
  apiToken: string | null;
  apiExecute: boolean;
  /** SearXNG instance for the `web_search` tool + deep_research. Defaults to
   *  the self-hosted homelab instance so web search works out of the box. */
  searxngUrl: string | null;
  /** agentlens instance for the `web_fetch` tool — token-efficient page
   *  extraction via `/parse?url=`. Defaults to the self-hosted homelab
   *  instance. When null (or unreachable), web_fetch falls back to a raw
   *  fetch + local HTML strip. */
  agentlensUrl: string | null;
  progressBar: boolean;
  modelStick: boolean;
  sync: { url: string; user: string; pass: string; auto: boolean } | null;
  rc: { enabled: boolean } | null;
  remote: { url: string; apiKey: string; allow?: string[] } | null;
  langfuse: { secretKey: string; publicKey: string; host?: string } | null;
  shellHistoryContext: boolean;
  fleet: Array<{ name: string; url: string }>;
  hooks: HooksConfig | null;
  /** MCP servers, keyed by name. Tools register as `[mcp:<name>]` source.
   *  See src/mcp.ts for transport details. Mirrors the Claude Desktop
   *  `mcpServers` shape so configs port directly. */
  mcpServers: Record<string, McpServerConfig> | null;
  /** Subagent (Task tool) constraints. Both fields optional — defaults
   *  preserve current behavior. */
  subagent: SubagentConfig | null;
  /** Language Server Protocol integration. Keyed by language label
   *  (e.g. "typescript", "go"). When null, LSP is fully disabled and the
   *  lsp_diagnostics tool is not registered. See docs/plans/v0.4-lsp.md. */
  lsp: Record<string, LspServerConfig> | null;
  /** Active extras (LazyVim-style language bundles). Each name corresponds
   *  to an entry in src/extras/builtins.ts. Adding via /extras add <name>
   *  installs the bundle's LSP recipes + hooks; the system-prompt section
   *  injects when cwd matches the bundle's projectMarkers. */
  extras: string[];
  /** When true, new sessions are stored in the JSONL tree-session format
   *  (one append-only file per session, supports `/tree` rewinds and labels).
   *  Existing legacy `.json` sessions remain readable either way. Default
   *  false until the new format has been dogfooded for a release cycle. */
  useJsonlSessions: boolean;
  /** Teacher-escalation self-learning: when a WEAK local student model fails a
   *  run, a strong teacher (DGX Qwen3.6-35B) distills a reusable skill into
   *  ~/.veepee-code/skills/ so the student succeeds next time. Off by default
   *  (opt-in — it calls the teacher endpoint on student failures). See
   *  src/teacher-escalation.ts. */
  teacher: { enabled: boolean; endpoint: string; model: string; apiKey?: string | null } | null;
}

export interface SubagentConfig {
  /** When set, the `task` tool rejects model names not in this list.
   *  Strongly recommended for fleets with pinned-per-server models so a
   *  typo can't trigger Ollama to pull/load an unintended model. Leave
   *  unset to allow any model the proxy will accept. */
  allowedModels?: string[];
  /** Override the hard-coded concurrent-subagent cap (default 4). Higher
   *  values risk vLLM slot exhaustion when subagents target the same
   *  server as the parent. */
  maxConcurrent?: number;
}

export type McpServerConfig =
  | {
      /** stdio transport — most common; spawns a child process. */
      command: string;
      args?: string[];
      env?: Record<string, string>;
      cwd?: string;
      allow?: string[];
      disabled?: boolean;
    }
  | {
      /** HTTP-based transport — `sse` by default for legacy configs. */
      url: string;
      transport?: 'sse' | 'http';
      headers?: Record<string, string>;
      allow?: string[];
      disabled?: boolean;
    };

/** Hooks configuration — see src/hooks.ts for runtime semantics. */
export interface HooksConfig {
  PreToolUse?: HookEntry[];
  PostToolUse?: HookEntry[];
  UserPromptSubmit?: HookEntry[];
  Stop?: HookEntry[];
  Notification?: HookEntry[];
  SessionStart?: HookEntry[];
  PreCompact?: HookEntry[];
  SubagentStop?: HookEntry[];
}

export interface HookEntry {
  /** Optional matcher — string (literal) or regex source — applied to the
   *  primary subject of the event (tool name for PreToolUse/PostToolUse,
   *  prompt text for UserPromptSubmit, etc.). When omitted, hook runs for
   *  every event. */
  matcher?: string;
  /** Shell command to run. Receives event JSON on stdin. Stdout is shown
   *  to the user as a system message; non-zero exit aborts the action
   *  (where applicable — e.g. PreToolUse can block the tool call). */
  command: string;
  /** Optional human-readable label shown in /hooks. */
  description?: string;
  /** Override default 30s timeout in milliseconds. */
  timeoutMs?: number;
  /** When false, the hook is configured but does not run. Lets users
   *  toggle a hook without removing it from settings. */
  enabled?: boolean;
}

export interface ConfigFile {
  /** Empty or null = no gateway (only valid with llmBackend "openai"). */
  proxyUrl?: string | null;
  llmBackend?: 'ollama' | 'openai';
  openaiBaseUrl?: string | null;
  openaiApiKey?: string | null;
  dashboardUrl?: string;
  model?: string | null;
  lockModel?: string | null;
  reviewModel?: string | null;
  secondModel?: string | null;
  /** Old name of secondModel. */
  planModel?: string | null;
  summarizerModel?: string | null;
  autoSwitch?: boolean;
  numCtx?: number | null;
  fallbackModels?: string[];
  mcpDeferAbove?: number;
  maxModelSize?: number;
  minModelSize?: number;
  apiPort?: number;
  apiHost?: string;
  apiToken?: string | null;
  apiExecute?: boolean;
  searxngUrl?: string | null;
  agentlensUrl?: string | null;
  progressBar?: boolean;
  modelStick?: boolean;
  sync?: { url: string; user: string; pass: string; auto: boolean } | null;
  rc?: { enabled: boolean } | null;
  remote?: { url: string; apiKey: string; allow?: string[] } | null;
  langfuse?: { secretKey: string; publicKey: string; host?: string } | null;
  shellHistoryContext?: boolean;
  fleet?: Array<{ name: string; url: string }>;
  hooks?: HooksConfig | null;
  mcpServers?: Record<string, McpServerConfig> | null;
  subagent?: SubagentConfig | null;
  lsp?: Record<string, LspServerConfig> | null;
  extras?: string[];
  useJsonlSessions?: boolean;
  teacher?: Config['teacher'];
}

const DEFAULTS: Config = {
  proxyUrl: 'http://localhost:11434',
  llmBackend: 'ollama',
  openaiBaseUrl: null,
  workerThinking: true,
  openaiApiKey: null,
  dashboardUrl: '',
  model: null,
  lockModel: null,
  reviewModel: null,
  secondModel: null,
  summarizerModel: null,
  autoSwitch: true,
  numCtx: null,
  fallbackModels: [],
  mcpDeferAbove: 12,
  maxModelSize: 40,
  minModelSize: 12,
  apiPort: 8484,
  apiHost: '127.0.0.1',
  apiToken: null,
  apiExecute: false,
  // No baked-in addresses: a fresh install on someone else's network must not
  // quietly call ours. Existing installs that relied on these got them written
  // into their .env by migrateToEnvFile().
  searxngUrl: null,
  agentlensUrl: null,
  progressBar: true,
  modelStick: false,
  sync: null,
  rc: null,
  remote: null,
  langfuse: null,
  shellHistoryContext: false,
  fleet: [],
  hooks: null,
  mcpServers: null,
  subagent: null,
  lsp: null,
  extras: [],
  useJsonlSessions: false,
  teacher: null,
};

// ─── Settings hierarchy paths ─────────────────────────────────────────
//
// Three layers, deeper wins. Mirrors Claude Code's settings layout:
//   1. Global   — ~/.veepee-code/settings.json
//   2. Project  — <cwd>/.veepee/settings.json (committed; team defaults)
//   3. Local    — <cwd>/.veepee/settings.local.json (gitignored; personal)

export type SettingsLayer = 'global' | 'project' | 'local';

export function getConfigDir(): string {
  return resolve(process.env.HOME || '~', '.veepee-code');
}

/** Canonical global settings file. New code writes here. */
export function getGlobalSettingsPath(): string {
  return resolve(getConfigDir(), 'settings.json');
}

/** Legacy filename — read for backward compat, migrated on first load. */
export function getLegacyGlobalSettingsPath(): string {
  return resolve(getConfigDir(), 'vcode.config.json');
}

export function getProjectSettingsDir(cwd: string = process.cwd()): string {
  return resolve(cwd, '.veepee');
}

export function getProjectSettingsPath(cwd: string = process.cwd()): string {
  return resolve(getProjectSettingsDir(cwd), 'settings.json');
}

export function getLocalSettingsPath(cwd: string = process.cwd()): string {
  return resolve(getProjectSettingsDir(cwd), 'settings.local.json');
}

/** Backward-compat alias. Returns the layer the global config currently
 *  lives at — `settings.json` if present, else legacy `vcode.config.json`. */
export function getConfigPath(): string {
  const newPath = getGlobalSettingsPath();
  if (existsSync(newPath)) return newPath;
  const legacy = getLegacyGlobalSettingsPath();
  if (existsSync(legacy)) return legacy;
  return newPath; // for write callers — they'll create settings.json
}

/** Returns the absolute path for a given settings layer. */
export function getSettingsPath(layer: SettingsLayer, cwd: string = process.cwd()): string {
  switch (layer) {
    case 'global': return getGlobalSettingsPath();
    case 'project': return getProjectSettingsPath(cwd);
    case 'local': return getLocalSettingsPath(cwd);
  }
}

// ─── Migrations ────────────────────────────────────────────────────────

// ─── .env: endpoints and secrets ──────────────────────────────────────
//
// ~/.veepee-code/.env holds where the models are and every credential;
// settings.json holds the structured rest. Each setting lives in exactly one
// of the two — a value in both would be two answers to one question.
//
// Precedence, lowest to highest: settings.json < .env < project settings <
// local settings < the process environment (VEEPEE_CODE_PROXY_URL=… vcode).

export function getEnvFilePath(): string {
  return resolve(getConfigDir(), '.env');
}

type EnvKeyPath = readonly [keyof ConfigFile] | readonly ['remote' | 'sync' | 'langfuse', string];

/** Every setting that lives in .env, and where it lands in the config. */
export const ENV_KEYS: ReadonlyArray<{ env: string; path: EnvKeyPath; secret?: boolean }> = [
  { env: 'VEEPEE_CODE_LLM_BACKEND', path: ['llmBackend'] },
  { env: 'VEEPEE_CODE_PROXY_URL', path: ['proxyUrl'] },
  { env: 'VEEPEE_CODE_OPENAI_BASE_URL', path: ['openaiBaseUrl'] },
  { env: 'VEEPEE_CODE_OPENAI_API_KEY', path: ['openaiApiKey'], secret: true },
  { env: 'VEEPEE_CODE_DASHBOARD_URL', path: ['dashboardUrl'] },
  { env: 'VEEPEE_CODE_API_TOKEN', path: ['apiToken'], secret: true },
  { env: 'SEARXNG_URL', path: ['searxngUrl'] },
  { env: 'AGENTLENS_URL', path: ['agentlensUrl'] },
  { env: 'VEEPEE_CODE_REMOTE_URL', path: ['remote', 'url'] },
  { env: 'VEEPEE_CODE_REMOTE_API_KEY', path: ['remote', 'apiKey'], secret: true },
  { env: 'VEEPEE_CODE_SYNC_URL', path: ['sync', 'url'] },
  { env: 'VEEPEE_CODE_SYNC_USER', path: ['sync', 'user'] },
  { env: 'VEEPEE_CODE_SYNC_PASS', path: ['sync', 'pass'], secret: true },
  { env: 'LANGFUSE_SECRET_KEY', path: ['langfuse', 'secretKey'], secret: true },
  { env: 'LANGFUSE_PUBLIC_KEY', path: ['langfuse', 'publicKey'] },
  { env: 'LANGFUSE_HOST', path: ['langfuse', 'host'] },
];

const NESTED = ['remote', 'sync', 'langfuse'] as const;

/** Turn env vars into a config layer. Only keys that are present count. */
export function envToConfig(vars: Map<string, string> | Record<string, string | undefined>): ConfigFile {
  const get = (k: string) => (vars instanceof Map ? vars.get(k) : vars[k]);
  const out: Record<string, unknown> = {};
  for (const { env, path } of ENV_KEYS) {
    const raw = get(env);
    if (raw === undefined) continue;
    if (path.length === 1) {
      const key = path[0];
      if (key === 'proxyUrl') out.proxyUrl = raw; // "" = no gateway
      else if (key === 'llmBackend') { if (raw === 'ollama' || raw === 'openai') out.llmBackend = raw; }
      else out[key] = raw === '' ? null : raw;
    } else if (raw !== '') {
      const [obj, field] = path;
      out[obj] = { ...(out[obj] as object | undefined), [field]: raw };
    }
  }
  return out as ConfigFile;
}

/** Split a config into its .env part (as env vars; null = remove) and the rest. */
export function splitEnvKeys(config: ConfigFile): { env: Record<string, string | null>; rest: ConfigFile } {
  const rest: Record<string, unknown> = { ...config };
  const env: Record<string, string | null> = {};
  for (const { env: name, path } of ENV_KEYS) {
    if (path.length === 1) {
      if (!(path[0] in config)) continue;
      const v = (config as Record<string, unknown>)[path[0]];
      delete rest[path[0]];
      if (v === undefined) continue;
      env[name] = v === null ? (path[0] === 'proxyUrl' ? '' : null) : String(v);
    } else {
      const [obj, field] = path;
      if (!(obj in config)) continue;
      const v = (config as Record<string, unknown>)[obj];
      if (v === null) { env[name] = null; continue; } // the whole block cleared
      if (v && typeof v === 'object' && field in v) {
        const fv = (v as Record<string, unknown>)[field];
        env[name] = fv === null || fv === undefined || fv === '' ? null : String(fv);
      }
    }
  }
  // What remains of a nested block once its env fields are taken out.
  for (const obj of NESTED) {
    const v = rest[obj];
    if (!v || typeof v !== 'object') continue;
    const fields = ENV_KEYS.filter(k => k.path[0] === obj).map(k => k.path[1] as string);
    const left = Object.fromEntries(Object.entries(v).filter(([k]) => !fields.includes(k)));
    if (Object.keys(left).length > 0) rest[obj] = left;
    else delete rest[obj];
  }
  return { env, rest: rest as ConfigFile };
}

/** Lay an env-derived layer over a config: nested blocks merge field by field,
 *  so VEEPEE_CODE_REMOTE_URL does not erase `remote.allow` from settings.json. */
function overlayEnv(base: ConfigFile, envLayer: ConfigFile): ConfigFile {
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(envLayer)) {
    if ((NESTED as readonly string[]).includes(k) && v && typeof v === 'object') {
      const prev = out[k];
      out[k] = { ...(prev && typeof prev === 'object' ? prev : {}), ...v };
    } else {
      out[k] = v;
    }
  }
  return out as ConfigFile;
}

/** The global layer as one config: settings.json with .env laid over it. */
export function readGlobalConfig(): ConfigFile {
  const settings = readConfigFileSafe(getGlobalSettingsPath());
  const global = Object.keys(settings).length > 0 ? settings : readConfigFileSafe(getLegacyGlobalSettingsPath());
  return overlayEnv(global, envToConfig(readEnvFile(getEnvFilePath())));
}

/**
 * Change global settings. Each key goes to its one file: endpoints and secrets
 * to .env, the rest to settings.json. `undefined` leaves a key alone, `null`
 * clears it.
 *
 * Pass only what changes. Callers used to spread a full loadConfig() into
 * saveConfigFile(), which wrote every default AND the project's local
 * overrides into the global file — run /model inside a repo with its own
 * proxyUrl and that proxy became everyone's.
 */
export function updateGlobalConfig(patch: ConfigFile): void {
  const { env, rest } = splitEnvKeys(patch);
  mkdirSync(getConfigDir(), { recursive: true });
  if (Object.keys(env).length > 0) updateEnvFile(getEnvFilePath(), env);

  const path = getGlobalSettingsPath();
  const current = readConfigFileSafeStrict(path);
  const next: Record<string, unknown> = { ...splitEnvKeys(current).rest };
  for (const [k, v] of Object.entries(rest)) {
    if (v === undefined) continue;
    if (v === null) { delete next[k]; continue; }
    const prev = next[k];
    next[k] = (NESTED as readonly string[]).includes(k) && prev && typeof prev === 'object' && typeof v === 'object'
      ? { ...prev, ...v }
      : v;
  }
  // A nested block cleared in the patch (remote: null) is cleared here too.
  for (const obj of NESTED) if ((patch as Record<string, unknown>)[obj] === null) delete next[obj];
  writeFileAtomicSync(path, JSON.stringify(next, null, 2) + '\n');
}

/** Old installs kept only a .env, including settings that are not endpoints. */
const LEGACY_ENV_SETTINGS: Record<string, (v: string) => [keyof ConfigFile, unknown]> = {
  VEEPEE_CODE_MODEL: (v) => ['model', v],
  VEEPEE_CODE_AUTO_SWITCH: (v) => ['autoSwitch', v !== 'false'],
  VEEPEE_CODE_MAX_MODEL_SIZE: (v) => ['maxModelSize', parseFloat(v)],
  VEEPEE_CODE_MIN_MODEL_SIZE: (v) => ['minModelSize', parseFloat(v)],
  VEEPEE_CODE_API_PORT: (v) => ['apiPort', parseInt(v, 10)],
  VEEPEE_CODE_API_HOST: (v) => ['apiHost', v],
  VEEPEE_CODE_API_EXECUTE: (v) => ['apiExecute', v === '1' || v === 'true'],
  VEEPEE_CODE_RC_ENABLED: (v) => ['rc', v === '1' || v === 'true' ? { enabled: true } : null],
};

/** The addresses DEFAULTS used to carry. An install that relied on one keeps
 *  it only if that server answers from this machine: on the network it was
 *  written for, behaviour is unchanged; anywhere else, a stranger's LAN
 *  address is not written into their config. */
const FORMER_DEFAULTS: Record<string, string> = {
  SEARXNG_URL: 'http://10.0.153.99:8888',
  AGENTLENS_URL: 'http://10.0.153.99:7001',
};

/**
 * One-time split into the two files. Returns true when it changed anything.
 *
 * - settings.json holding endpoints/secrets (every install before this) → they
 *   move to .env; settings.json is backed up first. A key already in .env wins.
 * - a .env holding non-endpoint settings (the oldest installs) → those move to
 *   settings.json. The .env itself is never renamed away any more.
 */
/** Does anything accept a TCP connection at this URL's host:port? Synchronous
 *  because config loading is; runs at most once per install (the migration). */
export function answers(url: string): boolean {
  let host: string, port: string;
  try {
    const u = new URL(url);
    host = u.hostname;
    port = u.port || (u.protocol === 'https:' ? '443' : '80');
  } catch {
    return false;
  }
  const probe = `const s=require('net').connect(${JSON.stringify(Number(port))},${JSON.stringify(host)},()=>process.exit(0));s.on('error',()=>process.exit(1));setTimeout(()=>process.exit(1),800)`;
  return spawnSync(process.execPath, ['-e', probe], { timeout: 2_000, stdio: 'ignore' }).status === 0;
}

export function migrateToEnvFile(): boolean {
  const envPath = getEnvFilePath();
  const settingsPath = getGlobalSettingsPath();
  const hadEnvFile = existsSync(envPath);
  const envVars = readEnvFile(envPath);
  let changed = false;

  const legacy = [...envVars.keys()].filter(k => k in LEGACY_ENV_SETTINGS);
  if (legacy.length > 0) {
    const moved: Record<string, unknown> = {};
    for (const k of legacy) {
      const [key, value] = LEGACY_ENV_SETTINGS[k](envVars.get(k)!);
      moved[key] = value;
    }
    const current = readConfigFileSafe(settingsPath);
    writeFileAtomicSync(settingsPath, JSON.stringify({ ...moved, ...current }, null, 2) + '\n');
    updateEnvFile(envPath, Object.fromEntries(legacy.map(k => [k, null])));
    changed = true;
  }

  if (!existsSync(settingsPath)) return changed;
  let settings: ConfigFile;
  try {
    settings = JSON.parse(readFileSync(settingsPath, 'utf-8'));
  } catch {
    return changed; // never rewrite a file we cannot read
  }
  const { env, rest } = splitEnvKeys(settings);
  const toWrite: Record<string, string> = {};
  for (const [k, v] of Object.entries(env)) {
    if (v !== null && !envVars.has(k)) toWrite[k] = v;
  }
  if (!hadEnvFile) {
    for (const [k, v] of Object.entries(FORMER_DEFAULTS)) {
      if (!(k in env) && !envVars.has(k) && answers(v)) toWrite[k] = v;
    }
  }
  // Only when something actually moves. A cleared block (`"sync": null`) counts
  // as an env key but leaves the file as it is, and it made this "one-time"
  // migration run, back up and announce itself on every launch.
  if (Object.keys(toWrite).length === 0 && JSON.stringify(rest) === JSON.stringify(settings)) return changed;

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  writeFileSync(`${settingsPath}.bak-envsplit-${stamp}`, readFileSync(settingsPath, 'utf-8'));
  if (Object.keys(toWrite).length > 0) updateEnvFile(envPath, toWrite);
  writeFileAtomicSync(settingsPath, JSON.stringify(rest, null, 2) + '\n');
  process.stderr.write(
    `\n  ▸ Config split: endpoints and secrets moved to ${envPath}\n` +
    `    Everything else stays in ${settingsPath} (backup: settings.json.bak-envsplit-${stamp})\n\n`,
  );
  return true;
}

/** Migrate legacy `vcode.config.json` to `settings.json`. Returns true if
 *  migration occurred. The legacy file is renamed to `vcode.config.json.bak`
 *  rather than deleted, so users can verify the migration succeeded.
 *
 *  Prints a one-time stderr notice on migration so users don't think their
 *  config was wiped when they see the renamed `.bak` and a new `settings.json`
 *  they don't recognize. (This was a real reported confusion — the migration
 *  used to be silent.)
 */
export function migrateLegacyConfig(): boolean {
  const newPath = getGlobalSettingsPath();
  const legacyPath = getLegacyGlobalSettingsPath();

  if (existsSync(newPath) || !existsSync(legacyPath)) return false;

  // Read, validate parse, rename.
  let content: string;
  try {
    content = readFileSync(legacyPath, 'utf-8');
    JSON.parse(content); // ensure it's valid JSON before migrating
  } catch {
    return false; // leave legacy in place if it's broken
  }

  writeFileSync(newPath, content);
  renameSync(legacyPath, legacyPath + '.bak');
  // Stderr so it doesn't pollute -p / --print stdout. Visible in interactive use.
  process.stderr.write(
    `\n  ▸ Config migrated: vcode.config.json → settings.json\n` +
    `    Your settings are intact at ${newPath}\n` +
    `    The old file was renamed to vcode.config.json.bak (safe to delete)\n\n`,
  );
  return true;
}

/** Startup must never fail on the split: a read-only config dir (a sandboxed
 *  service) keeps working, because un-moved keys in settings.json are still
 *  read — settings.json is the lowest layer, not an ignored one. */
function migrateToEnvFileSafely(): void {
  try {
    migrateToEnvFile();
  } catch (err) {
    process.stderr.write(`[VEEPEE Code] warning: could not move endpoints and secrets to .env (${err instanceof Error ? err.message : String(err)}); settings.json is still used as-is.\n`);
  }
}

// ─── Layered loading ───────────────────────────────────────────────────

function readConfigFileSafe(path: string): ConfigFile {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch (err) {
    // Corrupt file — log to stderr (visible in startup banner area) but
    // don't crash. Treat as empty layer; caller continues with other layers.
    process.stderr.write(`[VEEPEE Code] warning: could not parse ${path}: ${err instanceof Error ? err.message : String(err)}\n`);
    return {};
  }
}

/** Like readConfigFileSafe, but refuses a corrupt file instead of treating it
 *  as empty — a writer that read {} would overwrite every setting. */
function readConfigFileSafeStrict(path: string): ConfigFile {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, 'utf-8'));
  } catch (err) {
    throw new Error(`${path} is not valid JSON (${err instanceof Error ? err.message : String(err)}); fix it before changing settings`);
  }
}

/** Merge config layers — later layers override earlier. Shallow replacement
 *  on top-level fields (a project-level `remote` replaces global `remote`
 *  entirely; users redeclare to merge). */
function mergeLayers(...layers: ConfigFile[]): ConfigFile {
  const out: ConfigFile = {};
  for (const layer of layers) {
    for (const [k, v] of Object.entries(layer)) {
      if (v !== undefined) (out as Record<string, unknown>)[k] = v;
    }
  }
  return out;
}

export interface LoadedConfig extends Config {
  /** Per-layer raw contents for diagnostics (used by /settings show). */
  _layers?: { global: ConfigFile; project: ConfigFile; local: ConfigFile };
}

export function loadConfig(configPath?: string): Config {
  // If no explicit path, run migrations first
  if (configPath === undefined) {
    migrateLegacyConfig();
    migrateToEnvFileSafely();
  }

  let merged: ConfigFile = {};

  if (configPath !== undefined) {
    // Explicit path (used by tests). Empty string = skip file loading entirely.
    if (configPath) merged = readConfigFileSafe(configPath);
  } else {
    const project = readConfigFileSafe(getProjectSettingsPath());
    const local = readConfigFileSafe(getLocalSettingsPath());
    merged = overlayEnv(mergeLayers(readGlobalConfig(), project, local), envToConfig(process.env));
  }

  return {
    // null and "" both mean "no gateway"; only an absent key gets the default.
    proxyUrl: merged.proxyUrl === null ? '' : (merged.proxyUrl ?? DEFAULTS.proxyUrl),
    llmBackend: merged.llmBackend ?? DEFAULTS.llmBackend,
    openaiBaseUrl: merged.openaiBaseUrl ?? DEFAULTS.openaiBaseUrl,
    openaiApiKey: merged.openaiApiKey ?? DEFAULTS.openaiApiKey,
    dashboardUrl: merged.dashboardUrl ?? DEFAULTS.dashboardUrl,
    model: merged.model ?? DEFAULTS.model,
    lockModel: merged.lockModel ?? DEFAULTS.lockModel,
    reviewModel: merged.reviewModel ?? DEFAULTS.reviewModel,
    secondModel: merged.secondModel ?? merged.planModel ?? DEFAULTS.secondModel,
    summarizerModel: merged.summarizerModel ?? DEFAULTS.summarizerModel,
    autoSwitch: merged.autoSwitch ?? DEFAULTS.autoSwitch,
    numCtx: merged.numCtx ?? DEFAULTS.numCtx,
    mcpDeferAbove: typeof merged.mcpDeferAbove === 'number' ? merged.mcpDeferAbove : DEFAULTS.mcpDeferAbove,
    fallbackModels: Array.isArray(merged.fallbackModels) ? merged.fallbackModels.filter((m): m is string => typeof m === 'string' && !!m) : DEFAULTS.fallbackModels,
    maxModelSize: merged.maxModelSize ?? DEFAULTS.maxModelSize,
    minModelSize: merged.minModelSize ?? DEFAULTS.minModelSize,
    apiPort: merged.apiPort ?? DEFAULTS.apiPort,
    apiHost: merged.apiHost ?? DEFAULTS.apiHost,
    apiToken: merged.apiToken ?? DEFAULTS.apiToken,
    apiExecute: merged.apiExecute ?? DEFAULTS.apiExecute,
    searxngUrl: merged.searxngUrl ?? DEFAULTS.searxngUrl,
    agentlensUrl: merged.agentlensUrl ?? DEFAULTS.agentlensUrl,
    progressBar: merged.progressBar ?? DEFAULTS.progressBar,
    modelStick: merged.modelStick ?? DEFAULTS.modelStick,
    sync: merged.sync ?? DEFAULTS.sync,
    rc: merged.rc ?? DEFAULTS.rc,
    remote: merged.remote ?? DEFAULTS.remote,
    langfuse: merged.langfuse ?? DEFAULTS.langfuse,
    shellHistoryContext: merged.shellHistoryContext ?? DEFAULTS.shellHistoryContext,
    fleet: merged.fleet ?? DEFAULTS.fleet,
    hooks: merged.hooks ?? DEFAULTS.hooks,
    mcpServers: merged.mcpServers ?? DEFAULTS.mcpServers,
    subagent: merged.subagent ?? DEFAULTS.subagent,
    lsp: merged.lsp ?? DEFAULTS.lsp,
    extras: merged.extras ?? DEFAULTS.extras,
    useJsonlSessions: merged.useJsonlSessions ?? DEFAULTS.useJsonlSessions,
    teacher: merged.teacher ?? DEFAULTS.teacher,
  };
}

/** Load and return per-layer contents in addition to merged Config. Used by
 *  the /settings command to show provenance ("this value came from project"). */
export function loadConfigLayered(cwd: string = process.cwd()): LoadedConfig {
  migrateLegacyConfig();
  migrateToEnvFileSafely();
  const globalEffective = readGlobalConfig();
  const project = readConfigFileSafe(getProjectSettingsPath(cwd));
  const local = readConfigFileSafe(getLocalSettingsPath(cwd));
  const config = loadConfig() as LoadedConfig;
  config._layers = { global: globalEffective, project, local };
  return config;
}

/** Save configuration. Defaults to the global layer (preserves existing
 *  behavior). Pass `layer` to write to project or local instead. */
export function saveConfigFile(config: ConfigFile, layer: SettingsLayer = 'global'): void {
  if (layer === 'global') {
    // Endpoints and secrets belong in .env; never let them back into settings.json.
    const { env, rest } = splitEnvKeys(config);
    const present = Object.fromEntries(Object.entries(env).filter(([, v]) => v !== null)) as Record<string, string>;
    if (Object.keys(present).length > 0) updateEnvFile(getEnvFilePath(), present);
    writeFileAtomicSync(getGlobalSettingsPath(), JSON.stringify(rest, null, 2) + '\n');
    return;
  }
  const path = getSettingsPath(layer);
  // Project/local: ensure the parent dir exists.
  const dir = getProjectSettingsDir();
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  // Atomic: a truncated settings.json loses apiToken/lockModel and the next
  // start silently falls back to defaults.
  writeFileAtomicSync(path, JSON.stringify(config, null, 2) + '\n');
}

/** Read raw contents of a single settings layer (does not apply defaults). */
export function readSettingsLayer(layer: SettingsLayer, cwd: string = process.cwd()): ConfigFile {
  return readConfigFileSafe(getSettingsPath(layer, cwd));
}

// ─── .gitignore helper ─────────────────────────────────────────────────

/** Ensure `.veepee/settings.local.json` is gitignored in the project. Returns
 *  true if the gitignore was modified (caller can show a confirmation). */
export function ensureLocalSettingsGitignored(cwd: string = process.cwd()): boolean {
  const gitignorePath = resolve(cwd, '.gitignore');
  const ignoreLine = '.veepee/settings.local.json';
  let content = '';
  if (existsSync(gitignorePath)) {
    content = readFileSync(gitignorePath, 'utf-8');
    if (content.split('\n').some((l) => l.trim() === ignoreLine)) return false;
  }
  // Only auto-modify if .git exists — don't pollute non-git dirs.
  if (!existsSync(resolve(cwd, '.git'))) return false;
  const newContent = content + (content.endsWith('\n') || content === '' ? '' : '\n') +
    '\n# VEEPEE Code — local-only settings (personal overrides, not committed)\n' +
    ignoreLine + '\n';
  writeFileSync(gitignorePath, newContent);
  return true;
}

/** Convenience for callers that need just the project settings dir for
 *  related artifacts (commands/, hooks/, etc.). */
export { join as joinPath };
