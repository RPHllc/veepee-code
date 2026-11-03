/**
 * OpenAI /v1 chat client adapter.
 *
 * Lets vcode talk DIRECTLY to a vLLM (or any OpenAI-compatible) server's
 * documented `/v1/chat/completions` route, bypassing the Ollama-format
 * `/api/chat` + llm-gateway path. Opt-in via config `llmBackend: "openai"`.
 *
 * It exposes the SAME surface the agent consumes from the `ollama` client:
 * `.chat(params)` returns an async-iterable of Ollama-SHAPED chunks
 * (`{ message: { content?, tool_calls? }, eval_count?, prompt_eval_count? }`),
 * so the agent loop (src/agent.ts) needs no changes to how it reads the stream.
 *
 * Design notes:
 * - Qwen3.6 on vLLM (no reasoning parser) emits reasoning INLINE in `content`
 *   with a bare `</think>` close (empty-start-tag). We pass content through
 *   raw — the agent's existing orphan-`</think>` parser handles it.
 * - OpenAI streams tool-call `arguments` as a JSON *string*; the agent expects
 *   an *object* (Ollama shape). We accumulate + JSON.parse before yielding.
 * - The async generator ABORTS the underlying fetch in its `finally` block, so
 *   if the consumer stops iterating (user interrupt, turn end) the streaming
 *   request is cancelled — never orphaned (orphaned streams can wedge vLLM).
 */

/**
 * Marker for a tool call whose arguments arrived truncated.
 *
 * Exported because the registry has to recognise it: a silently-defaulted call
 * is indistinguishable from an intended one, which is the failure this exists to
 * stop.
 */
export const TRUNCATED_ARGS_KEY = '__vcode_truncated_arguments';

interface OllamaChunk {
  message: { content?: string; thinking?: string; tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }> };
  eval_count?: number;
  prompt_eval_count?: number;
  /** Prompt tokens served from the KV prefix cache, when the server reports it. */
  prompt_cached_count?: number;
  eval_duration?: number;
  done?: boolean;
}

interface ChatParams {
  model: string;
  messages: unknown[];
  tools?: unknown[];
  think?: boolean;
  /** Same contract as the Ollama client: a stream only when `true`, otherwise
   *  one collected response. The agent loop always passes `true`; subagents,
   *  compaction and one-shot prompts do not, and used to get the gateway. */
  stream?: boolean;
  /** Abort signal from the agent's run controller. Wired to the underlying
   *  fetch so a stall-timeout / user interrupt cancels the HTTP request
   *  immediately (a stalled stream never delivers a chunk, so the consumer's
   *  loop-level abort check can't run — the signal is the only thing that
   *  prevents an orphaned, potentially server-wedging stream). */
  signal?: AbortSignal;
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    presence_penalty?: number;
    repeat_penalty?: number;
    num_ctx?: number;
    num_predict?: number;
    seed?: number;
  };
}

/** The non-streaming Ollama response shape (`message` plus final counters). */
export interface OllamaResponse {
  message: { role: 'assistant'; content: string; thinking?: string; tool_calls?: NonNullable<OllamaChunk['message']['tool_calls']> };
  done: true;
  eval_count?: number;
  prompt_eval_count?: number;
  eval_duration?: number;
}

/** Fold a chunk stream into one response, the way Ollama answers `stream: false`. */
async function collectResponse(stream: AsyncIterable<OllamaChunk>): Promise<OllamaResponse> {
  let content = '';
  let thinking = '';
  const toolCalls: NonNullable<OllamaChunk['message']['tool_calls']> = [];
  const out: OllamaResponse = { message: { role: 'assistant', content: '' }, done: true };
  for await (const chunk of stream) {
    content += chunk.message.content ?? '';
    thinking += chunk.message.thinking ?? '';
    if (chunk.message.tool_calls) toolCalls.push(...chunk.message.tool_calls);
    if (chunk.eval_count !== undefined) out.eval_count = chunk.eval_count;
    if (chunk.prompt_eval_count !== undefined) out.prompt_eval_count = chunk.prompt_eval_count;
    if (chunk.eval_duration !== undefined) out.eval_duration = chunk.eval_duration;
  }
  out.message.content = content;
  if (thinking) out.message.thinking = thinking;
  if (toolCalls.length > 0) out.message.tool_calls = toolCalls;
  return out;
}

/**
 * Translate vcode's Ollama-shaped message history to the OpenAI /v1 shape.
 * The /v1 schema strictly requires: assistant `tool_calls` carry an `id`,
 * their `arguments` are a JSON *string*, and each following `tool` result
 * carries a matching `tool_call_id`. vcode stores none of these (Ollama is
 * lax), so we synthesize ids and match tool results to the tool_calls that
 * preceded them.
 *
 * Matching is BY TOOL NAME first, not blind position: the parallel read-only
 * execution path can store tool results out of call order (denied/blocked
 * calls are appended after executed ones), so a positional FIFO would map a
 * result to the wrong tool_call_id. We keep a per-name FIFO queue of ids and
 * shift by the result's `tool_name`; a global FIFO is the fallback when a
 * result carries no name (older messages) so the request still validates.
 */
export function toOpenAIMessages(messages: any[]): any[] {
  const out: any[] = [];
  const byName = new Map<string, string[]>();   // tool name -> queued ids (in call order)
  const globalQueue: string[] = [];             // fallback FIFO for un-named results
  let counter = 0;
  for (const m of messages) {
    if (m.role === 'assistant' && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
      const tool_calls = m.tool_calls.map((tc: any) => {
        const id = `call_${counter++}`;
        const name = tc.function?.name ?? '';
        if (!byName.has(name)) byName.set(name, []);
        byName.get(name)!.push(id);
        globalQueue.push(id);
        const a = tc.function?.arguments;
        const argsStr = typeof a === 'string' ? a : JSON.stringify(a ?? {});
        return { id, type: 'function', function: { name, arguments: argsStr } };
      });
      out.push({ role: 'assistant', content: m.content ?? '', tool_calls });
    } else if (m.role === 'tool') {
      const name: string | undefined = m.tool_name;
      let id: string | undefined;
      if (name && byName.get(name)?.length) {
        id = byName.get(name)!.shift();
        // keep the global queue consistent
        const gi = globalQueue.indexOf(id!);
        if (gi >= 0) globalQueue.splice(gi, 1);
      } else {
        id = globalQueue.shift();
        // also drop it from its per-name queue if present
        if (id) for (const q of byName.values()) { const i = q.indexOf(id); if (i >= 0) { q.splice(i, 1); break; } }
      }
      out.push({ role: 'tool', tool_call_id: id ?? `call_orphan_${counter++}`, content: m.content ?? '' });
    } else {
      out.push(m);
    }
  }
  return out;
}

export class OpenAIChatClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(baseUrl: string, apiKey?: string) {
    // Normalize: accept ".../v1" or bare host; ensure single "/v1" suffix.
    let b = baseUrl.replace(/\/+$/, '');
    if (!/\/v1$/.test(b)) b = b + '/v1';
    this.baseUrl = b;
    this.apiKey = apiKey;
  }

  async chat(params: ChatParams & { stream: true }): Promise<AsyncIterable<OllamaChunk>>;
  async chat(params: ChatParams): Promise<OllamaResponse>;
  async chat(params: ChatParams): Promise<AsyncIterable<OllamaChunk> | OllamaResponse> {
    const o = params.options || {};
    const body: Record<string, unknown> = {
      model: params.model,
      messages: toOpenAIMessages(params.messages as any[]),
      stream: true,
      stream_options: { include_usage: true },
      // Standard OpenAI sampling params
      temperature: o.temperature,
      top_p: o.top_p,
      presence_penalty: o.presence_penalty,
      max_tokens: o.num_predict,
      // vLLM-honored extras (ignored by servers that don't support them)
      top_k: o.top_k,
      min_p: o.min_p,
      repetition_penalty: o.repeat_penalty,
      // Qwen3 thinking toggle: think:false -> enable_thinking=false
      chat_template_kwargs: { enable_thinking: params.think !== false },
    };
    if (o.seed !== undefined) body.seed = o.seed;
    if (params.tools && params.tools.length > 0) {
      body.tools = params.tools;
      body.tool_choice = 'auto';
    }

    // `controller` aborts the fetch when the consumer stops iterating (the
    // generator's finally). The agent's run signal (params.signal) aborts it
    // on stall-timeout / user interrupt. Combine both so EITHER cancels the
    // real HTTP request — a stalled stream is otherwise never cancelled.
    const controller = new AbortController();
    const fetchSignal = params.signal
      ? AbortSignal.any([params.signal, controller.signal])
      : controller.signal;
    const headers: Record<string, string> = { 'content-type': 'application/json' };
    if (this.apiKey) headers['authorization'] = `Bearer ${this.apiKey}`;

    const startedAt = Date.now();
    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: fetchSignal,
    });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '');
      throw new Error(`OpenAI /v1 backend HTTP ${res.status}: ${text.slice(0, 300)}`);
    }

    const stream = this.parse(res.body, controller, startedAt);
    return params.stream === true ? stream : collectResponse(stream);
  }

  private async *parse(body: ReadableStream<Uint8Array>, controller: AbortController, startedAt: number): AsyncGenerator<OllamaChunk> {
    const decoder = new TextDecoder();
    // Accumulate tool calls by index across delta frames.
    const toolAcc = new Map<number, { name: string; args: string }>();
    let buf = '';
    let sawToolCalls = false;
    let lastUsage: {
      prompt_tokens?: number;
      completion_tokens?: number;
      /** vLLM reports how much of the prompt was served from the KV prefix
       *  cache. Nothing read it, which is why a 0%-hit-rate prompt layout went
       *  undiagnosed for a day while everything else was being measured. */
      prompt_tokens_details?: { cached_tokens?: number };
    } | null = null;
    try {
      for await (const chunk of body as AsyncIterable<Uint8Array>) {
        buf += decoder.decode(chunk, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf('\n')) >= 0) {
          let line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          if (line.startsWith('data:')) line = line.slice(5).trim();
          if (line === '[DONE]') continue;
          let j: any;
          try { j = JSON.parse(line); } catch { continue; }

          // usage frame (from stream_options.include_usage). Capture the
          // LAST-seen usage and emit once after the loop — some servers send a
          // running total per chunk, and the agent SUMS eval_count across
          // chunks, so emitting inline would over-count.
          if (j.usage) lastUsage = j.usage;

          const choice = j.choices?.[0];
          if (!choice) continue;
          const delta = choice.delta || {};

          if (typeof delta.content === 'string' && delta.content.length > 0) {
            yield { message: { content: delta.content } };
          }
          // Some servers surface reasoning on its own channel. It is yielded as
          // `thinking`, NOT folded into `content` — the agent renders that
          // collapsed and keeps it out of the assistant message.
          //
          // FOLDING IT INTO CONTENT WAS WORSE THAN DROPPING IT. Measured on the
          // DGX 2026-08-23 for the prompt "hi, are you ready to code?":
          // reasoning = 1050 chars, content = 97 chars. Folded, the user saw the
          // entire chain of thought printed above the answer — and, worse, every
          // heuristic in the agent that reads the assistant's own words saw it
          // too. Reasoning says "Let me check…" / "I need to…" almost by
          // definition, which is exactly the STATED_INTENT pattern
          // `shouldForceAct` fires on, so a plain greeting earned two
          // force-act nudges and two pointless bash calls, and the model then
          // narrated the hidden [SYSTEM] nudge back to the user.
          //
          // THE FIELD NAME IS SERVER-SPECIFIC AND GETTING IT WRONG IS SILENT.
          // vLLM 0.23.1 (the DGX Spark, --reasoning-parser deepseek_r1) emits
          // `delta.reasoning`; it NEVER emits `reasoning_content`. Reading only
          // the latter meant that whenever the model routed its answer into the
          // reasoning channel, vcode received ZERO text and printed nothing —
          // measured 2026-08-07 against the live DGX: with enable_thinking=true
          // (vcode's default, `think: params.think !== false`) a plain prompt
          // returned content=0 chars and reasoning=849 chars. A real
          // `vcode -p "find and fix a bug"` run against that engine read nine
          // files, ran four turns, and emitted ONE BYTE of output.
          // Tool calls were never affected — they arrive in `delta.tool_calls` —
          // which is why the agent looked like it was "doing nothing" while
          // actually working. Accept both spellings; servers send one or the
          // other, never both.
          const reasoningDelta =
            typeof delta.reasoning_content === 'string' && delta.reasoning_content.length > 0
              ? delta.reasoning_content
              : typeof delta.reasoning === 'string' && delta.reasoning.length > 0
                ? delta.reasoning
                : '';
          if (reasoningDelta) {
            yield { message: { thinking: reasoningDelta } };
          }

          if (Array.isArray(delta.tool_calls)) {
            sawToolCalls = true;
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0;
              const cur = toolAcc.get(idx) || { name: '', args: '' };
              if (tc.function?.name) cur.name = tc.function.name;
              if (typeof tc.function?.arguments === 'string') cur.args += tc.function.arguments;
              toolAcc.set(idx, cur);
            }
          }
        }
      }

      // Emit accumulated tool calls as a single Ollama-shaped chunk with
      // arguments PARSED to objects (the agent expects objects, not strings).
      //
      // A stream truncated mid-arguments used to become `{}` on the reasoning
      // that "the tool's own arg validation surfaces a clear error". That holds
      // only for tools with required fields. For a tool whose arguments are all
      // optional — `list_files`, `git status` — an empty object is a VALID call,
      // so a truncated request silently executed as a defaulted one and the
      // model was told it succeeded. Absence must be its own value: the marker
      // makes every tool reject it, and the registry says why.
      if (sawToolCalls && toolAcc.size > 0) {
        const tool_calls = [...toolAcc.entries()]
          .sort((a, b) => a[0] - b[0])
          .map(([, v]) => {
            let args: Record<string, unknown> = {};
            try {
              args = v.args ? JSON.parse(v.args) : {};
            } catch {
              args = { [TRUNCATED_ARGS_KEY]: v.args.slice(0, 200) };
            }
            return { function: { name: v.name, arguments: args } };
          });
        yield { message: { tool_calls }, done: true };
      }

      // Final metrics chunk: token counts + wall-clock duration (ns) so the
      // agent's tokens/sec calc (needs eval_duration) produces a real number.
      if (lastUsage) {
        yield {
          message: {},
          prompt_eval_count: lastUsage.prompt_tokens,
          eval_count: lastUsage.completion_tokens,
          eval_duration: (Date.now() - startedAt) * 1e6,
          prompt_cached_count: lastUsage.prompt_tokens_details?.cached_tokens,
        };
      }
    } finally {
      // If the consumer stopped iterating early, cancel the request so the
      // server-side generation is aborted rather than orphaned.
      try { controller.abort(); } catch { /* noop */ }
    }
  }
}
