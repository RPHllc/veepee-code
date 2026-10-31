/** Isolated headless entry point. Does not load desktop/project settings or plugins. */
import { readFileSync, statSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { z } from 'zod';
import { loadConfig } from './config.js';
import { WorkerPolicy } from './worker-policy.js';
import { ToolRegistry } from './tools/registry.js';
import { registerWebTools } from './tools/web.js';
import { ModelManager } from './models.js';
import { PermissionManager } from './permissions.js';
import { Agent } from './agent.js';
import { startApiServer } from './api.js';

const settingsSchema = z.object({
  openaiBaseUrl: z.string().url(),
  model: z.string().min(1).max(200),
  searxngUrl: z.string().url(),
  apiHost: z.enum(['127.0.0.1', '0.0.0.0']).default('127.0.0.1'),
  apiPort: z.number().int().min(1024).max(65535).default(8484),
}).strict();

export async function startWorker(env: NodeJS.ProcessEnv = process.env) {
  const policy = WorkerPolicy.load(env.CONDUCTOR_WORKER_POLICY_PATH || '');
  // Only installed, audited implementations are eligible in this entry point.
  policy.validateAvailable(['web_search']);
  if (!env.VCODE_WORKER_SETTINGS || !env.VCODE_API_TOKEN_FILE) throw new Error('Worker settings and token file are required');
  const settings = settingsSchema.parse(JSON.parse(readFileSync(env.VCODE_WORKER_SETTINGS, 'utf8')));
  const tokenPath = env.VCODE_API_TOKEN_FILE;
  const stat = statSync(tokenPath);
  if (!stat.isFile() || (stat.mode & 0o077)) throw new Error('API token file must be owner-only');
  const token = readFileSync(tokenPath, 'utf8').trim();
  if (token.length < 32 || token.length > 256 || /\s/.test(token)) throw new Error('Invalid API token');
  for (const endpoint of [settings.openaiBaseUrl, settings.searxngUrl]) {
    const url = new URL(endpoint);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('Invalid worker endpoint');
  }
  const config = { ...loadConfig(''), ...settings, llmBackend: 'openai' as const,
    proxyUrl: '', lockModel: settings.model, autoSwitch: false, apiToken: token,
    numCtx: 16384, agentlensUrl: null, fallbackModels: [],
  };
  const registry = new ToolRegistry(policy);
  for (const tool of registerWebTools(config)) if (policy.allows(tool.name)) registry.register(tool);
  policy.validateAvailable(registry.names());
  const models = new ModelManager(config);
  await models.discover();
  models.switchTo(settings.model);
  const agent = new Agent(config, registry, models, new PermissionManager());
  const api = startApiServer({ port: settings.apiPort, host: settings.apiHost,
    agent, registry, modelManager: models, apiToken: token });
  await api.ready;
  return { api, agent, registry };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  startWorker().then(({ api, agent }) => {
    console.log('Research worker starting (authenticated, policy enforced)');
    for (const signal of ['SIGTERM', 'SIGINT'] as const) process.once(signal, () => {
      agent.abort(); api.close();
      setTimeout(() => process.exit(0), 2000).unref();
    });
  }).catch(() => { console.error('Worker startup failed: invalid or unavailable protected configuration'); process.exitCode = 1; });
}
