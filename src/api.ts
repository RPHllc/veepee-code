import { ResearchJobs } from './research-jobs.js';
import { z } from 'zod';
import { readBody, BodyIntakeTimeoutError, WORKER_BODY_TIMEOUT_MS } from './api-body.js';
import type { WorkerReadiness } from './worker-readiness.js';
import { createServer, type IncomingMessage, type ServerResponse } from 'http';
import type { Agent, AgentEvent } from './agent.js';
import { AgentBusyError } from './agent.js';
import type { ModelManager } from './models.js';
import type { ToolRegistry } from './tools/registry.js';
import type { Config } from './config.js';
import { escalateAndLearn, type TeacherConfig } from './teacher-escalation.js';
import { safeTokenEquals } from './auth.js';

interface ApiConfig {
  port: number;
  host?: string; // bind address (default: 127.0.0.1)
  agent: Agent;
  modelManager: ModelManager;
  registry: ToolRegistry;
  apiToken?: string | null; // optional auth token; if set, all requests must include it
  workerReadiness?: () => Promise<WorkerReadiness>;
  apiExecute?: boolean; // enable /api/execute endpoint
  rcEnabled?: boolean; // enable Remote Connect (wider CORS, 0.0.0.0 bind)
  rcRequestHandler?: (req: IncomingMessage, res: ServerResponse, url: URL) => Promise<boolean>;
  teacher?: Config['teacher']; // teacher-escalation config (student-failure → DGX distills a skill)
}

/** Build the teacher config for escalateAndLearn, or null when disabled. */
function teacherFrom(t: Config['teacher'] | undefined): TeacherConfig | null {
  if (!t || !t.enabled || !t.endpoint || !t.model) return null;
  return { enabled: true, endpoint: t.endpoint, model: t.model, apiKey: t.apiKey ?? null };
}

/**
 * OpenAI-compatible API server that allows other tools (Claude Code, Gemini CLI,
 * Codex, etc.) to use VEEPEE Code as a backend with full tool access.
 *
 * Endpoints:
 *   POST /v1/chat/completions  — OpenAI-compatible chat (with tool execution)
 *   GET  /v1/models            — List available models
 *   GET  /api/tools            — List available tools
 *   GET  /api/status           — Session status
 *   POST /api/execute          — Execute a specific tool directly
 */
export function startApiServer(config: ApiConfig): { port: number; connectionCount: number; close: () => void; ready: Promise<void> } {
  const { agent, modelManager, registry } = config;

  const apiToken = config.apiToken || null;
  const researchJobs = registry.policy ? new ResearchJobs(agent, registry) : null;
  if (registry.policy) {
    if (!apiToken || apiToken.length < 32) throw new Error('Worker requires a 32+ character API token');
    if (config.apiExecute || config.rcEnabled || config.rcRequestHandler || config.teacher?.enabled) {
      throw new Error('Worker forbids direct execution, Remote Connect and teacher side effects');
    }
  }

  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    // CORS — allow any origin when RC is enabled, localhost only otherwise
    const origin = req.headers.origin || '';
    if (config.rcEnabled) {
      res.setHeader('Access-Control-Allow-Origin', origin || '*');
    } else if (origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1')) {
      res.setHeader('Access-Control-Allow-Origin', origin);
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS' && !registry.policy) {
      res.writeHead(204);
      res.end();
      return;
    }

    // Auth check — if token is configured, require Bearer token
    //
    // Exempt: GET /rc (the login page), GET /rc/stream (carries its own token as
    // a query param, since EventSource cannot set headers), and the login
    // page's own static assets. The assets are the empty app shell and hold no
    // data — but without them the browser cannot render the screen on which the
    // token is entered, so gating them makes the UI unreachable by design.
    // Everything that reads or changes state stays behind the check.
    const reqPath = req.url?.split('?')[0] || '';
    const RC_PUBLIC = new Set(['/rc', '/rc/', '/rc/app.js', '/rc/app.css', '/rc/stream']);
    const skipAuth = !registry.policy && req.method === 'GET' && RC_PUBLIC.has(reqPath);
    if (apiToken && !skipAuth) {
      const authHeader = req.headers.authorization || '';
      const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
      if (!safeTokenEquals(token, apiToken)) {
        sendJson(res, 401, { error: 'Unauthorized — set Authorization: Bearer <token>' });
        return;
      }
    }

    const url = new URL(req.url || '/', `http://localhost:${config.port}`);
    const path = url.pathname;

    try {
      // RC routes — handle first if enabled
      if (config.rcRequestHandler) {
        const handled = await config.rcRequestHandler(req, res, url);
        if (handled) return;
      }

      if (researchJobs && path === '/v1/research/status' && req.method === 'GET') {
        sendJson(res, 200, researchJobs.status()); return;
      }
      if (researchJobs && path === '/v1/research/jobs' && req.method === 'POST') {
        const result = researchJobs.submit(JSON.parse(await readBody(req, WORKER_BODY_TIMEOUT_MS)));
        sendJson(res, result.code, result.body); return;
      }
      const researchMatch = /^\/v1\/research\/jobs\/([a-f0-9-]{36})(\/cancel)?$/.exec(path);
      if (researchJobs && researchMatch) {
        if (req.method === 'GET' && !researchMatch[2]) {
          const job = researchJobs.get(researchMatch[1]); sendJson(res, job ? 200 : 404, job ?? { error: 'Receipt not found' }); return;
        }
        if (req.method === 'POST' && researchMatch[2]) {
          const parsed = z.object({ worker_instance: z.string().uuid(), reason: z.enum(['cancelled', 'timed_out']) }).strict().safeParse(JSON.parse(await readBody(req, WORKER_BODY_TIMEOUT_MS)));
          if (!parsed.success) { sendJson(res, 422, { error: 'Invalid cancellation' }); return; }
          const result = researchJobs.cancel(researchMatch[1], parsed.data.worker_instance, parsed.data.reason);
          sendJson(res, result.code, result.body); return;
        }
      }
      if (path === '/ready'  && req.method === 'GET' && registry.policy && config.workerReadiness) {
        const readiness = await config.workerReadiness();
        sendJson(res, readiness.ready ? 200 : 503, readiness);
        return;
      }

      // OpenAI-compatible chat completions
      if (path === '/v1/chat/completions' && req.method === 'POST') {
        const body = await readBody(req, registry.policy ? WORKER_BODY_TIMEOUT_MS : undefined);
        const data = JSON.parse(body) as {
          model?: string;
          messages: Array<{ role: string; content: string }>;
          stream?: boolean;
          tools?: unknown[];
        };

        // Use specified model or current default
        if (data.model && registry.policy && data.model !== modelManager.getCurrentModel()) {
          sendJson(res, 400, { error: 'Worker model is pinned' });
          return;
        }
        if (data.model) {
          const profile = modelManager.getProfile(data.model);
          if (profile) {
            agent.setModel(data.model);
          }
        }

        // Extract the last user message
        const lastUserMsg = [...data.messages].reverse().find(m => m.role === 'user');
        if (!lastUserMsg || typeof lastUserMsg.content !== 'string' || (registry.policy && lastUserMsg.content.length > 8192)) {
          sendJson(res, 400, { error: 'No user message found' });
          return;
        }

        // If client provides tool definitions, constrain agent to those tools only
        const userContent = lastUserMsg.content;
        let clientToolNames: string[] | null = null;
        if (data.tools && Array.isArray(data.tools) && data.tools.length > 0) {
          clientToolNames = data.tools
            .map((t: any) => t?.function?.name)
            .filter(Boolean) as string[];
        }
        if (registry.policy) {
          // Empty client list means no tools; omission retains only the server ceiling.
          clientToolNames = registry.policy.intersect(data.tools === undefined ? null :
            Array.isArray(data.tools) ? data.tools.map((t: any) => t?.function?.name).filter((n: unknown) => typeof n === 'string') : []);
          if (agent.isRunning()) { sendBusy(res); return; }
          agent.clear();
          const timer = setTimeout(() => { agent.abort(); res.destroy(); }, registry.policy.deadlineMs);
          const cleanup = () => { clearTimeout(timer); if (!res.writableFinished) agent.abort(); };
          res.once('close', cleanup);
          res.once('finish', () => clearTimeout(timer));
        }
        if (data.stream) {
          // Claim the agent BEFORE committing response headers — once we've
          // written a 200 + SSE headers we can no longer report a 409.
          let stream: AsyncGenerator<AgentEvent>;
          try {
            stream = agent.run(userContent, {
              permissionMode: 'auto_allow',
              allowedTools: clientToolNames,
            });
          } catch (err) {
            if (err instanceof AgentBusyError) {
              sendBusy(res);
              return;
            }
            throw err;
          }

          // Streaming response (SSE)
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });

          const model = modelManager.getCurrentModel();
          let toolCallIndex = -1;
          // teacher-escalation: assemble the run outcome as events stream by
          const _teacher = teacherFrom(config.teacher);
          let _assistantText = '';
          const _outcomeTools: Array<{ name: string; success: boolean; error?: string }> = [];
          const _outcomeErrors: string[] = [];
          let _outcomeStuck = false;

          for await (const event of stream) {
            if (event.type === 'text' && event.content) {
              if (_teacher) _assistantText += event.content;
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{
                  index: 0,
                  delta: { content: event.content },
                  finish_reason: null,
                }],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (event.type === 'tool_call') {
              // Emit standard OpenAI tool_calls delta
              toolCallIndex++;
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{
                  index: 0,
                  delta: {
                    tool_calls: [{
                      index: toolCallIndex,
                      id: `call_${Date.now()}_${toolCallIndex}`,
                      type: 'function',
                      function: {
                        name: event.name,
                        arguments: JSON.stringify(event.args || {}),
                      },
                    }],
                  },
                  finish_reason: null,
                }],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (event.type === 'tool_result') {
              if (_teacher) _outcomeTools.push({ name: event.name || '', success: event.success !== false, error: event.success === false ? String(event.error || event.content || '') : undefined });
              // Tool results aren't part of OpenAI streaming spec, but useful for clients
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{
                  index: 0,
                  delta: { content: '' },
                  finish_reason: null,
                }],
                veepee_code: {
                  tool_result: {
                    name: event.name,
                    success: event.success,
                    output: (event.content || '').slice(0, 1000),
                  },
                },
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (event.type === 'done') {
              const chunk = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model,
                choices: [{
                  index: 0,
                  delta: {},
                  finish_reason: 'stop',
                }],
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
              res.write('data: [DONE]\n\n');
            } else if (event.type === 'error' && registry.policy) {
              res.write(`data: ${JSON.stringify({ error: 'Worker inference/run failed', code: 'WORKER_RUN_FAILED' })}\n\n`);
            } else if (event.type === 'error' && _teacher) {
              const em = String(event.error || event.content || '');
              _outcomeErrors.push(em);
              if (/stuck|no progress|no output|repeat/i.test(em)) _outcomeStuck = true;
            }
          }

          res.end();
          // teacher-escalation: fire-and-forget after the client has the full
          // response — a weak student's failure → the DGX distills a skill.
          if (_teacher) {
            escalateAndLearn(
              { userMessage: userContent, studentModel: model, assistantText: _assistantText, toolCalls: _outcomeTools, errors: _outcomeErrors, stuck: _outcomeStuck },
              _teacher,
            ).then((r) => { if (r.escalated) console.error(`[teacher] ${r.reason} -> ${r.skillPath}`); })
             .catch((e) => console.error('[teacher] escalation error:', e?.message || e));
          }
          return;
        }

        // Non-streaming response
        let result;
        try {
          result = await agent.runSync(userContent, {
            permissionMode: 'auto_allow',
            allowedTools: clientToolNames,
          });
        } catch (err) {
          if (err instanceof AgentBusyError) {
            sendBusy(res);
            return;
          }
          throw err;
        }
        if (registry.policy && result.errors.length) {
          sendJson(res, 502, { error: 'Worker inference/run failed', code: 'WORKER_RUN_FAILED' });
          return;
        }
        const model = modelManager.getCurrentModel();

        // Build standard OpenAI tool_calls array
        const toolCallsMsg = result.toolCalls.length > 0
          ? result.toolCalls.map((tc, i) => ({
              id: `call_${Date.now()}_${i}`,
              type: 'function' as const,
              function: { name: tc.name, arguments: JSON.stringify(tc.args) },
            }))
          : undefined;

        sendJson(res, 200, {
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: result.content,
              ...(toolCallsMsg ? { tool_calls: toolCallsMsg } : {}),
            },
            finish_reason: 'stop',
          }],
          ...(registry.policy ? { veepee_code: { tool_results: result.toolCalls.map(t => ({ name: t.name, success: t.success, output: t.result.slice(0, 4096) })) } } : {}),
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
        });
        // teacher-escalation (non-streaming): distill a skill if the student failed.
        {
          const _teacher = teacherFrom(config.teacher);
          if (_teacher) {
            escalateAndLearn(
              {
                userMessage: userContent,
                studentModel: model,
                assistantText: result.content,
                toolCalls: result.toolCalls.map((t) => ({ name: t.name, success: t.success, error: t.success ? undefined : t.result })),
                errors: result.errors,
                stuck: result.stuck,
              },
              _teacher,
            ).then((r) => { if (r.escalated) console.error(`[teacher] ${r.reason} -> ${r.skillPath}`); })
             .catch((e) => console.error('[teacher] escalation error:', e?.message || e));
          }
        }
        return;
      }

      // List models (OpenAI-compatible)
      if (path === '/v1/models' && req.method === 'GET') {
        const models = modelManager.getAllModels();
        sendJson(res, 200, {
          object: 'list',
          data: models.map(m => ({
            id: m.name,
            object: 'model',
            created: Math.floor(Date.now() / 1000),
            owned_by: 'ollama',
            capabilities: m.capabilities,
            tier: m.tier,
            parameter_size: m.parameterSize,
            score: m.score,
            is_loaded: m.isLoaded,
          })),
        });
        return;
      }

      // List tools
      if (path === '/api/tools' && req.method === 'GET') {
        const tools = registry.list();
        sendJson(res, 200, {
          tools: tools.map(t => ({
            name: t.name,
            description: t.description,
          })),
          count: tools.length,
        });
        return;
      }

      // Execute a tool directly — requires VEEPEE_CODE_API_EXECUTE=1 opt-in
      if (path === '/api/execute' && req.method === 'POST') {
        if (registry.policy || !config.apiExecute) {
          sendJson(res, 403, { error: '/api/execute is disabled. Set apiExecute to true in ~/.veepee-code/settings.json to enable.' });
          return;
        }

        const body = await readBody(req, registry.policy ? WORKER_BODY_TIMEOUT_MS : undefined);
        const data = JSON.parse(body) as { tool: string; args: Record<string, unknown> };

        if (!data.tool) {
          sendJson(res, 400, { error: 'tool is required' });
          return;
        }

        const result = await registry.execute(data.tool, data.args || {});
        sendJson(res, 200, result);
        return;
      }

      // Session status
      if (path === '/api/status' && req.method === 'GET') {
        sendJson(res, 200, {
          model: modelManager.getCurrentModel(),
          tools: registry.count(),
          messages: agent.getContext().messageCount(),
          tokens_estimate: agent.getContext().estimateTokens(),
          cwd: process.cwd(),
        });
        return;
      }

      // Health check
      if (path === '/' || path === '/health') {
        sendJson(res, 200, { status: 'ok', service: 'veepee-code' });
        return;
      }

      sendJson(res, 404, { error: 'Not found' });
    } catch (err) {
      if (err instanceof BodyIntakeTimeoutError) {
        res.setHeader('Connection', 'close');
        res.once('finish', () => req.destroy());
        sendJson(res, 408, { error: 'Request body intake timed out' });
        return;
      }
      const msg = err instanceof Error ? err.message : String(err);
      sendJson(res, 500, { error: registry.policy ? 'Worker request failed' : msg });
    }
  });

  // Port hunting is bounded: an unbounded retry walks past 65535 and then spins
  // on EACCES/ERANGE forever. Any other bind error (EACCES, EADDRNOTAVAIL) used
  // to be swallowed silently, leaving a dead server nobody noticed — log it.
  let readyResolve!: () => void;
  let readyReject!: (err: Error) => void;
  const ready = new Promise<void>((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
  void ready.catch(() => undefined);
  server.once('listening', readyResolve);
  const MAX_PORT_ATTEMPTS = 20;
  let portAttempts = 0;
  server.on('error', (err: NodeJS.ErrnoException) => {
    if (!registry.policy && err.code === 'EADDRINUSE' && portAttempts < MAX_PORT_ATTEMPTS && config.port < 65535) {
      portAttempts++;
      config.port++;
      const bindHost = config.rcEnabled ? '0.0.0.0' : (config.host || '127.0.0.1');
      server.listen(config.port, bindHost);
      return;
    }
    readyReject(err);
    console.error(`[api] server error (${err.code || 'unknown'}): ${err.message} — API server is not listening`);
  });

  // Track live external connections so the TUI can hide the API indicator when
  // nothing is actually connected. HTTP keep-alive and RC streams hold sockets
  // open, so a non-zero count maps to "an external client is using us".
  let connectionCount = 0;
  server.on('connection', (socket) => {
    connectionCount++;
    socket.once('close', () => { connectionCount--; });
  });

  const bindHost = config.rcEnabled ? '0.0.0.0' : (config.host || '127.0.0.1');
  server.listen(config.port, bindHost);

  return {
    ready,
    get port() {
      const address = server.address();
      return typeof address === 'object' && address ? address.port : config.port;
    },
    get connectionCount() {
      return connectionCount;
    },
    close: () => server.close(),
  };
}

/** 409 for "a run is already in flight" — the shared agent takes one at a time. */
function sendBusy(res: ServerResponse): void {
  res.setHeader('Retry-After', '1');
  sendJson(res, 409, {
    error: 'agent busy — a run is already in progress',
    code: 'AGENT_BUSY',
    retry_after_ms: 1000,
  });
}

function sendJson(res: ServerResponse, status: number, data: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}
