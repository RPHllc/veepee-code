import { Ollama } from 'ollama';
import type { Message, ToolCall } from 'ollama';
import type { Config } from './config.js';
import { OpenAIChatClient } from './openai-adapter.js';
import { answerText } from './llm-answer.js';
import { parseTextToolCalls, TextToolCallGate } from './text-tool-calls.js';
import { ollamaNumCtx, vllmMaxModelLen } from './ollama-context.js';
import { TodoList, buildTodoTool } from './todo.js';
import type { ToolResult } from './tools/types.js';
import { retryDecision, isTransportFailure, describeError } from './retry.js';
import { killRunningBashCommands } from './tools/coding.js';
import type { ToolRegistry } from './tools/registry.js';
import type { PermissionManager } from './permissions.js';
import type { BenchmarkResult } from './benchmark.js';
import { ContextManager, CHAT_TOOLS } from './context.js';
import { ModelManager } from './models.js';
import type { ModelRoster } from './benchmark.js';
import { SubAgentManager } from './subagent.js';
import { runHooks, shouldBlock, type HookExecResult, type HookEventName } from './hooks.js';
import { report as reportAgentState } from './agentstate.js';
import { previewEdit, previewWrite } from './diff.js';

import type { CheckpointManager } from './checkpoint.js';
import { signatureOf, callSignatureOf, detectStuckSignature, detectRepeatedFailure, LOOP_WINDOW, LOOP_MAX_REPEATS, REPEATED_FAILURE_LIMIT, detectContentRepetition, CONTENT_REPETITION_LIMIT, type SignedStep2 } from './loop-detection.js';
import { generationLimiter } from './generation-limit.js';
import type { PermissionPosture } from './permissions.js';
import { readFile, readFile as readFileAsync } from 'node:fs/promises';
import { resolve, relative } from 'path';
import { existsSync, readFileSync } from 'fs';
import { createChatClient, isDirectOnly } from './llm-client.js';

export interface AgentEvent {
  type: 'text' | 'tool_call' | 'tool_result' | 'model_switch' | 'thinking' | 'info' | 'done' | 'error' | 'permission_denied' | 'reset_stream' | 'hook_output';
  content?: string;
  name?: string;
  args?: Record<string, unknown>;
  success?: boolean;
  error?: string;
  sources?: string[];
  from?: string;
  to?: string;
  // Token metrics from Ollama (available on 'done' events)
  evalCount?: number;      // tokens generated
  promptEvalCount?: number; // prompt tokens processed
  tokensPerSecond?: number; // actual generation speed
  /** Prompt tokens the server served from its KV prefix cache, and the share of
   *  the prompt that represents. A low share means the prompt layout is
   *  re-prefilling work the server already had — the dominant turn-latency cost
   *  on a long session, and invisible until it is reported. */
  promptCachedCount?: number;
  promptCachedShare?: number;
  // Hook metadata (available on 'hook_output' events)
  hookEvent?: HookEventName;
  hookLayer?: 'global' | 'project' | 'local';
  hookExitCode?: number;
  hookBlocked?: boolean;
}

/**
 * Sampling presets for Qwen3-family models.
 * Source: https://huggingface.co/Qwen/Qwen3.6-35B-A3B "Best Practices" section.
 * Forwarded by ollama_proxy → vLLM (see proxy/vllm-adapter.ts mapOptionsToVllm).
 * Note: Qwen docs say `repetition_penalty`; Ollama-shape key is `repeat_penalty`
 * which the proxy renames back to `repetition_penalty` for vLLM.
 *
 * CODING: thinking-mode coding tasks (act/plan) — tighter sampling.
 * INSTRUCT: non-thinking conversational mode (chat) — Qwen's "Instruct" preset.
 *           Used when `think: false` is honored by the proxy (Qwen3 + vLLM only).
 */
/**
 * Sampling for act/plan — where all the long structured generation happens.
 *
 * ⚠ `presence_penalty` was 0.0 here while the CHAT preset below used 1.5, and
 * that asymmetry cost a real task: asked to write a 15-record JSON fixture, the
 * model emitted "Still writing game data... " **1,341 times**, produced 43KB of
 * output, wrote no files, and would have run to the deadline. Qwen's own
 * guidance recommends a presence penalty specifically to prevent endless
 * repetition, and calls it out for quantized weights — which is exactly what the
 * DGX serves (NVFP4).
 *
 * 1.0 rather than chat's 1.5: the penalty discourages reusing tokens already
 * present, and code legitimately repeats itself far more than prose does —
 * `const`, `return`, an identifier used twelve times in a function. Too high
 * degrades the code to avoid the very tokens it needs.
 *
 * Behind `VCODE_NO_PRESENCE_PENALTY=1` so it can be A/B'd against itself on one
 * binary, per the same-build rule.
 */
export const QWEN_CODING_PRESET = {
  temperature: 0.6,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: process.env.VCODE_NO_PRESENCE_PENALTY === '1' ? 0.0 : 1.0,
  repeat_penalty: 1.0,
} as const;

export const QWEN_INSTRUCT_PRESET = {
  temperature: 0.7,
  top_p: 0.80,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  repeat_penalty: 1.0,
} as const;

/** What makes a reply a plan worth keeping across compaction: a plan heading,
 *  or explicit Step/Phase numbering. A numbered bold list is NOT one — that
 *  pattern saved a four-line answer listing two held commands as "the plan". */
export const PLAN_CONTENT_PATTERNS = [
  /^#{1,3}\s+(implementation|action)\s+plan/im,
  /^#{1,3}\s+plan\b/im,
  /^#{2,3}\s+(step|phase)\s+\d/im,
  /(?:^|\n)(?:step|phase)\s+\d+[.:]/im,
];

export function isPlanContent(content: string): boolean {
  return !!content && content.length >= 200 && PLAN_CONTENT_PATTERNS.some(p => p.test(content));
}

/** Prefix for a plan restored after compaction. Reference, never an order: the
 *  old wording ("immediately execute the next incomplete step without waiting
 *  for user input") turned any stale list into unapproved actions. */
export const RESTORED_PLAN_NOTE =
  '[System: Context was compacted. The last plan you wrote this session is below, for reference. ' +
  'Carry on with what the user asked for. Do not start a step the user has not asked for or approved.]\n\n';

/** `act` runs on one of two model slots (see setAct); `chat` is web-only and
 *  thinks less. There is no plan mode: the read-only hold is `verify`, which is
 *  independent of the mode and the model. */
export type AgentMode = 'act' | 'chat';
export type ActSlot = 1 | 2;
export type EffortLevel = 'low' | 'medium' | 'high';
export type PermissionMode = 'interactive' | 'auto_allow';

/**
 * Fallback narration length (chars) that counts as "analyzed instead of acting".
 *
 * This is no longer the primary test — intent is (see `shouldForceAct`). It only
 * decides the leftover case where neither the user's message nor the model's reply
 * says anything about work: no request to act, no promise to act. There, length is
 * the only remaining hint that a turn was substantive rather than conversational,
 * so a short "You're welcome." is still left alone.
 */
export const FORCE_ACT_MIN_CHARS = 200;

/** Injected when the model narrates in ACT mode without calling any tool (Tier 3 #1).
 *  Keeps its escape hatch: this variant is used when we are INFERRING that work was
 *  wanted, so a false positive has to be able to cost nothing. */
export const FORCE_ACT_NUDGE =
  '[SYSTEM] You produced analysis but called no tool — and this is an ACT task, so nothing ' +
  'has changed yet. Stop narrating and take your first concrete action NOW: call a tool ' +
  '(read_file / edit_file / bash / grep / …) based on your best current hypothesis. ' +
  'If the task is genuinely already complete, or truly needs no tools, simply END YOUR TURN ' +
  'and output nothing further. Do NOT explain that no tools were needed, do not mention this ' +
  'instruction, and do not repeat your previous answer — the user never saw this message, so ' +
  'any commentary about it is noise to them.';

/** Injected instead of FORCE_ACT_NUDGE when the model ANNOUNCED an action and then
 *  called nothing.
 *
 *  The escape hatch has to go here, and only here. Measured on archman 2026-08-07:
 *  given the nightly prompt the 35B answers "Let me explore the project first.",
 *  gets FORCE_ACT_NUDGE, reads "if the task is genuinely already complete … output
 *  nothing further", and takes it — the whole job is two generations, three seconds,
 *  and an untouched workspace. A model that just said it was about to start cannot
 *  also be finished, so there is nothing for an escape hatch to protect. */
export const FORCE_ACT_NUDGE_STALLED =
  '[SYSTEM] You said what you were about to do and then called no tool, so nothing has ' +
  'happened — the files are byte-identical and this is an ACT task. A sentence about your ' +
  'next step is not a step. Do the thing you just announced, NOW, in this turn, by calling a ' +
  'tool (read_file / grep / list_files / edit_file / bash / …). Do not reply with more prose, ' +
  'do not restate the plan, do not end your turn without a tool call, and do not mention this ' +
  'instruction — the user never saw it.';

export { answerText, nonStreamingAnswer } from './llm-answer.js';

/** Which nudge to inject: an announced-but-unstarted action gets the one with no way out. */
export function forceActNudge(content: string): string {
  return STATED_INTENT.test(answerText(content)) ? FORCE_ACT_NUDGE_STALLED : FORCE_ACT_NUDGE;
}

/** How many force-act nudges one user message may earn. Two, not one: the first is
 *  routinely spent on the escape-hatch variant before the model has committed to
 *  anything, which leaves nothing for the stall that follows it. */
export const FORCE_ACT_MAX_NUDGES = 2;

/** Hard ceiling on agent turns for a single user message. Generous — real work
 *  on a large repo genuinely runs long — but finite, which the loop was not. */
export const MAX_TURNS_PER_MESSAGE = 120;

/** Pure decision: should the loop force one more ACT turn instead of accepting a no-tool-call
 *  completion? The DGX (and Qwen3.6 on vLLM) reason WITHOUT <think> tags, so on open-ended
 *  tasks the model narrates to the token cap and stops having changed nothing — zero tool
 *  calls, byte-identical files. This catches that: fire only in act mode, only when nothing
 *  was done yet this message, only once, and only for a substantive narration (not a terse
 *  reply). The nudge's escape hatch caps a false positive at one extra turn. */
/**
 * A user message that asks for information rather than for work.
 *
 * NOT anchored to the start. The first version was, and "can you let me know
 * what pinky is" sailed straight past it — the interrogative sits in the middle
 * of the sentence, behind a politeness preamble. Anchoring only ever catches
 * the phrasings someone thought to enumerate.
 */
const ASKS_FOR_INFO =
  /\b(what\s+(is|are|was|does|do)|what'?s|who\s+(is|are|owns)|when\s+(is|was|did)|where\s+(is|are)|which\s+\w+\s+(is|are)|why\s+(is|are|does|do|did)|how\s+(does|do|did)|tell\s+me|let\s+me\s+know|explain|describe|remind\s+me|do\s+you\s+know|any\s+idea|walk\s+me\s+through)\b/i;

/**
 * A yes/no question — "is the DGX up", "does newsfeed use pgvector".
 *
 * Anchored, unlike the pattern above, because these auxiliaries are far too
 * common mid-sentence to be a signal anywhere else ("check if the DGX is up"
 * is an instruction, and contains "is").
 */
const YES_NO_QUESTION = /^\s*(is|are|was|were|does|do|did|has|have|can|could|should|will|would)\s+\w/i;

/**
 * A user message asking for work to be done.
 *
 * Checked alongside the above because the two co-occur: "explain why the test
 * fails and fix it" both asks and instructs, and the nudge should still fire.
 * An imperative anywhere wins.
 *
 * "make sense" is carved out. It is not a request to make anything — "does that
 * make sense?" is the most ordinary thing a user says mid-conversation, and once
 * a work request fires the nudge at any length (see `shouldForceAct`), letting
 * that phrase count as an imperative nudges a plain "Yes, exactly."
 */
const ASKS_FOR_WORK =
  /\b(fix|add|implement|create|write|refactor|update|remove|delete|rename|migrate|build|install|run|debug|change|make(?!\s+sense\b)|set\s+up|clean\s+up|commit|deploy|test\s+the|check\s+(if|whether|that))\b/i;

/**
 * The model announcing work it has not done — the thing this nudge exists for.
 *
 * "I'll start by reading …", "Let me check …", "First I need to …". A narration
 * that states intent and then calls no tool is a turn where nothing happened.
 * An ANSWER has no such marker: it describes what is, not what it is about to do.
 */
// The apostrophes are REQUIRED on "I'll" and "let's". Without them the pattern
// matches the ordinary words "ill" and "lets" — and the answer that exposed
// this bug contains "a plain-markdown index that LETS any AI assistant…", which
// made every suppression test fail on its first run.
// "let me know" is excluded. It is the model handing the turn BACK ("let me know if
// you'd like me to change it"), the opposite of announcing work, and it closes a large
// share of otherwise-finished answers. It only started to matter once a stated intent
// could fire the nudge after the model had already done its work.
const STATED_INTENT =
  /\b(i['’]ll\b|i\s+will\b|let\s+me\b(?!\s+know\b)|i['’]m\s+going\s+to\b|i\s+need\s+to\b|i\s+should\b|we\s+should\b|let['’]s\b|first,?\s+i\b|next,?\s+i\b|start\s+by\b|going\s+to\s+(check|look|read|run|search))/i;

/** Pure decision: should the loop force one more ACT turn instead of accepting a no-tool-call
 *  completion? The DGX (and Qwen3.6 on vLLM) reason WITHOUT <think> tags, so on open-ended
 *  tasks the model narrates to the token cap and stops having changed nothing — zero tool
 *  calls, byte-identical files. This catches that: fire only in act mode, only when nothing
 *  was done yet this message, only once, and only for a substantive narration (not a terse
 *  reply). The nudge's escape hatch caps a false positive at one extra turn.
 *
 *  Length alone was not enough. "what is pinky" earned a correct 400-character
 *  answer needing no tools, tripped the length test, got nudged, and the model
 *  — having already answered — invented work: it read three unrelated files out
 *  of the current directory and summarised them. A good answer to a question is
 *  long, so length is a proxy for "substantive", not for "a task went undone".
 *
 *  So a lookup request that produced no stated intent to act is now left alone.
 *  Both conditions are required to suppress: "why is this test failing" is a
 *  lookup in form, but if the model answers it with "let me check the logs" and
 *  no tool call, that is exactly the stall worth nudging.
 *
 *  The length floor USED to run first, and that made it the real gate: nothing
 *  shorter than 200 characters could ever be nudged, whatever it said. The 35B's
 *  failure mode moved — it stopped narrating to the token cap and started stalling
 *  in one line. The Nightly Engineer's five barren nights (2026-08-03..07) are
 *  literally these stored job results: "Let me explore the project first." (33 B)
 *  and "Let me explore the project first to find a real issue to fix." (61 B), each
 *  a ~2s run with zero tool calls and a byte-identical workspace. Both are pure
 *  STATED_INTENT — the exact signal this function already computes — and both were
 *  thrown away by a length test applied before anyone looked at them.
 *
 *  So intent leads and length is the fallback. A promise to act, or a user who
 *  plainly asked for work, is a stall at ANY length (including an empty reply,
 *  which is a stall by definition). Length only decides the residual case where
 *  neither side mentioned work at all — a plain conversational exchange — which is
 *  what the floor was actually protecting.
 */
export function shouldForceAct(opts: {
  mode: AgentMode; hasActedThisMessage: boolean; alreadyForced: boolean; content: string;
  /** The user's message. Optional so existing callers keep compiling; when absent
   *  only the model's own stated intent (and the length fallback) apply. */
  userMessage?: string;
}): boolean {
  if (opts.mode !== 'act') return false;
  if (opts.alreadyForced) return false;

  const msg = opts.userMessage ?? '';
  const askedForWork = ASKS_FOR_WORK.test(msg);
  const askedForInfo = (ASKS_FOR_INFO.test(msg) || YES_NO_QUESTION.test(msg)) && !askedForWork;
  // The ANSWER, never the reasoning that preceded it — see `answerText`.
  const answer = answerText(opts.content);
  const promisedAction = STATED_INTENT.test(answer);

  // Once the model HAS acted, a closing summary is a real completion and must be
  // left alone — that veto is most of what keeps this guard safe. The single
  // exception is a promise of MORE work: "Let me look at the grading executor more
  // closely" after 198 seconds of reading and zero edits (archman, 2026-08-07) is
  // the same stall as before, just later in the turn.
  if (opts.hasActedThisMessage) return promisedAction;

  // A question answered without any promise to act is an ANSWER, not a stall.
  // This is the a88545a/744569a fix and it stays first — it must win over the
  // work/length tests below, or "what is pinky" starts inventing work again.
  if (askedForInfo && !promisedAction) return false;

  // Either side referring to work makes this a task turn that produced nothing.
  // Terseness is not evidence of completion here — it is how the stall looks.
  if (promisedAction || askedForWork) return true;

  // Nobody mentioned work. Only a substantive narration is worth a nudge; a short
  // conversational reply is left alone.
  return answer.length >= FORCE_ACT_MIN_CHARS;
}

/**
 * Commands that cannot verify anything, however successfully they run.
 *
 * The guard cannot tell `npm test` from `ls`, and it never will from a string.
 * But it can refuse to accept the obviously-inert ones, which is what a model
 * reaches for when it is narrating progress rather than checking it.
 */
const NON_VERIFYING_BASH = /^\s*(ls|pwd|echo|cat|head|tail|which|whoami|date|find|wc|sleep|true|clear|cd|export|env|printenv|git\s+(status|log|diff|show|branch|remote))\b/;

/** Did this bash command plausibly verify the change that preceded it? */
export function bashVerifies(command: string): boolean {
  const segments = String(command).split(/[;&|]+|\n/).map(s => s.trim()).filter(Boolean);
  return segments.some(seg => !NON_VERIFYING_BASH.test(seg));
}

/** Tools that change code (leave the workspace in a state that ought to be verified). */
export const CODE_MUTATION_TOOLS = new Set(['write_file', 'edit_file', 'multi_edit']);

/** Injected when the model finishes a code change without ever running it (Tier: daily-driver #1).
 *  This is the self-repair trigger — it drives generate → run → fix. */
export const FORCE_VERIFY_NUDGE =
  '[SYSTEM] You changed code but never ran it — so you do not actually know it works, and vcode ' +
  'confidently shipping a broken edit is exactly the failure to avoid. Before finishing: run the ' +
  'project tests, a build/typecheck, or execute the code (via bash), read the output, and FIX ' +
  'anything that fails. If there is genuinely no way to run it here, say what you would run and why ' +
  'you cannot, in one sentence.';

/** Pure decision: force a verify-and-fix turn instead of accepting completion. Fires in act
 *  mode when code was changed and nothing has been run since the last edit, at most once. This
 *  is the harness half of self-repair; the model figures out HOW to verify (it knows the project's
 *  test/build command better than a hardcoded guess). Escape hatch caps a false positive at one turn. */
export function shouldForceVerify(opts: {
  mode: AgentMode; codeChangedUnverified: boolean; alreadyForced: boolean;
}): boolean {
  return opts.mode === 'act' && opts.codeChangedUnverified && !opts.alreadyForced;
}

/**
 * Thrown when a run is requested while another is already in flight on the
 * same Agent. Callers that own an HTTP response should surface this as 409
 * before committing any response headers.
 */


/** After this many failed attempts on an unreachable model, move to a fallback. */
const FALLBACK_AFTER_ATTEMPTS = 2;
/** How long a model that failed is skipped before being tried again. */
const MODEL_DOWN_MS = 2 * 60_000;

export class AgentBusyError extends Error {
  readonly code = 'AGENT_BUSY';
  constructor(message = 'agent busy — a run is already in progress') {
    super(message);
    this.name = 'AgentBusyError';
  }
}

export class Agent {
  /** Trusted harness telemetry; never exposes reasoning text or grants permissions. */
  private telemetry?: (event: Record<string, number | string>) => void;
  setTelemetry(sink?: (event: Record<string, number | string>) => void): void { this.telemetry = sink; }
  private trace(event: Record<string, number | string>): void { try { this.telemetry?.(event); } catch { /* diagnostics cannot authorize or interrupt work */ } }
  private ollama: Ollama;
  private context: ContextManager;
  private modelManager: ModelManager;
  private registry: ToolRegistry;
  private permissions: PermissionManager;
  /** Optional file checkpointing. Best-effort: never blocks or fails a turn. */
  private checkpoints: CheckpointManager | null = null;
  /** True once this turn has taken its snapshot, so we take at most one. */
  private checkpointedThisTurn = false;
  private optimalContextSizes = new Map<string, number>();
  /** Models whose window has already been asked for (answered or not). */
  private contextProbed = new Set<string>();
  private mode: AgentMode = 'act';
  /** Which model act mode runs on: 1 = the primary, 2 = the second model. */
  private actSlot: ActSlot = 1;
  /** The slot-1 model and its auto-switch setting, recorded EVERY time Act 1
   *  is left — so a /model choice made in Act 1 is what Act 1 returns to. */
  private primaryModel: string | null = null;
  private primaryAutoSwitch: boolean | null = null;
  /** The last plan written this session, restored after compaction. Never on disk. */
  private sessionPlan: string | null = null;
  /** Read-only until approved: edits, shell and subagents are refused until the
   *  user approves a proposal via request_approval. Off by default. */
  private verify = false;
  private previousModel: string | null = null;
  private roster: ModelRoster | null = null;
  private subAgents: SubAgentManager;
  private config: Config;
  private abortController: AbortController | null = null;
  /** True while a run is in flight. Guards the shared context + abort handle. */
    private runLock = false;
    /** Captured when _run() completes — lets callers inspect what changed. */
    private _runResult: { editedPaths: Set<string>; codeChangedUnverified: boolean } | null = null;

    /** Snapshot of what this run changed. Available after run() completes. */
    getRunResult(): { editedPaths: Set<string>; codeChangedUnverified: boolean } | null {
      return this._runResult;
    }
  private effort: EffortLevel = 'medium';
  private modelStick = false; // when true, mode switches don't change the model
  /** How much the user wants to be asked. Cycled with Shift+Tab in the TUI. */
  private posture: PermissionPosture = 'manual';
  private openaiBackend = false; // true when using the OpenAIChatClient adapter
  /** Models the direct (openai-backend) endpoint actually serves. Empty when
   *  the gateway is the primary transport, since it fronts the whole fleet. */
  private directModels = new Set<string>();
  private sessionStarted = false;
  /** How many tokens the server's prompt count has exceeded our estimate by
   *  (learned from an overflow error); added to every later estimate. */
  private promptUndercount = 0;
  /** Models that just failed at the transport level, and until when to skip
   *  them (see pickModel). */
  private modelDownUntil = new Map<string, number>();
  /** The direct server's connection failed: route everything via the gateway until then. */
  private directDownUntil = 0;
  /** Models that produced a response this session, in first-use order. */
  private modelsUsed = new Set<string>();
  /** Human-readable record of every fallback taken (for -p callers' reports). */
  private fallbackNotes: string[] = [];

  getModelsUsed(): string[] { return [...this.modelsUsed]; }
  getFallbackNotes(): string[] { return [...this.fallbackNotes]; }
  /** This agent's task list (todo_write) — see todo.ts. */
  readonly todos = new TodoList();
  private readonly todoTool = buildTodoTool(this.todos);
  private todoNudged = false;
  private directModelsListed = false;
  /** Models whose server refused `think` ("does not support thinking"). A
   *  plain Ollama rejects the whole request rather than ignoring the flag, so
   *  every non-reasoning model failed on the first prompt. Learned on the
   *  first refusal, then never sent again for that model. */
  private noThinkModels = new Set<string>();
  /** Lazily-built gateway client, used for models the direct endpoint lacks. */
  private gatewayClient: Ollama | null = null;

  constructor(config: Config, registry: ToolRegistry, modelManager: ModelManager, permissions: PermissionManager) {
    // Backend transport: "openai" talks straight to a vLLM /v1 server
    // (bypassing the llm-gateway + Ollama format); "ollama" (default) uses the
    // gateway. The adapter is duck-compatible with the Ollama `.chat()` surface
    // the agent loop consumes, so it's cast to the same field type.
    if (config.llmBackend === 'openai' && config.openaiBaseUrl) {
      this.ollama = new OpenAIChatClient(config.openaiBaseUrl, config.openaiApiKey ?? undefined) as unknown as Ollama;
      this.openaiBackend = true;
      // Until the server's own list arrives (ensureDirectModels), assume it
      // serves the primary. Every other model (reviewModel, subagent models)
      // lives behind the gateway. See clientFor().
      const primary = config.lockModel ?? config.model;
      if (primary) this.directModels.add(primary);
    } else {
      this.ollama = new Ollama({ host: config.proxyUrl, headers: { "x-ollama-source": "vcode" } });
    }
    this.context = new ContextManager(undefined, !!registry.policy);
    // The task list is this agent's own (API sessions each get an Agent but
    // share one registry); the registry only carries the tool's schema.
    if (!registry.policy && !registry.has(this.todoTool.name)) registry.register(buildTodoTool(new TodoList()));
    this.context.setExtraVolatile(() => this.todos.contextBlock());
    this.modelManager = modelManager;
    this.registry = registry;
    this.permissions = permissions;
    this.config = config;
    // Subagent manager is always available — initialized with null roster
    // so the `task` tool can use it before benchmark roster loads. Roster
    // will replace it once available (see loadRoster). Importantly, the
    // SAME instance persists, so registered tools holding a reference
    // continue to work after roster swap.
    this.subAgents = new SubAgentManager(this.config, this.registry, null);
    // Subagents get the same dangerous-command gate as the parent loop.
    this.subAgents.setPermissions(permissions);
    // A subagent of a working agent can always run where its parent runs.
    this.subAgents.setDefaultModel(modelManager.getCurrentModel());

    if (!registry.policy) {
      this.loadBenchmarkContextSizes();
      this.loadRoster();
    }
  }

  /** Load the model roster from benchmark results */
  private async loadRoster(): Promise<void> {
    const rosterPath = resolve(process.env.HOME || '~', '.veepee-code', 'benchmarks', 'roster.json');
    if (!existsSync(rosterPath)) return;
    try {
      const data = await readFile(rosterPath, 'utf-8');
      this.roster = JSON.parse(data) as ModelRoster;
      // Update IN PLACE. This used to re-instantiate, one line under a comment
      // promising the instance persists — so `createTaskTool` and
      // `setOnTransition`, both of which capture the manager synchronously at
      // startup, kept the old object while `/agents` queried the new one.
      this.subAgents.setRoster(this.roster);
    } catch { /* ignore */ }
  }

  getMode(): AgentMode {
    return this.mode;
  }

  getRoster(): ModelRoster | null {
    return this.roster;
  }

  getModelStick(): boolean {
    return this.modelStick;
  }

  setModelStick(on: boolean): void {
    this.modelStick = on;
    if (on) {
      this.modelManager.setAutoSwitch(false);
    }
  }

  /**
   * The posture in force for a turn.
   *
   * `auto_allow` is the legacy option name used by the eval harness, goal mode
   * and `-p`, all of which run unattended — it maps to `auto`. Otherwise the
   * agent's own posture applies, which the TUI cycles with Shift+Tab.
   */
  private postureFor(mode: PermissionMode): PermissionPosture {
    return mode === 'auto_allow' ? 'auto' : this.posture;
  }

  getPosture(): PermissionPosture {
    return this.posture;
  }

  setPosture(posture: PermissionPosture): void {
    this.posture = posture;
  }

  getActSlot(): ActSlot {
    return this.actSlot;
  }

  getVerify(): boolean {
    return this.verify;
  }

  setVerify(on: boolean): void {
    this.verify = on;
    this.permissions.setVerify(on); // subagents check through the manager too
    this.context.setVerify(on);
  }

  /** The model each act slot runs on, or null when slot 2 has none. */
  slotModel(slot: ActSlot): string | null {
    if (slot === 1) return this.primaryModel ?? this.modelManager.getCurrentModel();
    return this.config.secondModel ?? this.config.fallbackModels[0] ?? null;
  }

  /**
   * Act on slot 1 (the primary model) or slot 2 (the second model).
   *
   * The second model is `secondModel`, else the first fallback model. It is
   * switched to by name and needs no discovered profile: under lockModel
   * nothing else is discovered, and a non-primary model routes via the gateway
   * (see clientFor). Returns null when slot 2 has no model configured.
   */
  setAct(slot: ActSlot): { model: string } | null {
    const target = this.slotModel(slot);
    if (!target) return null;
    // Already there: nothing to switch, and nothing to undo (a /model choice stays).
    if (this.mode === 'act' && this.actSlot === slot) return { model: this.modelManager.getCurrentModel() };
    this.leavingAct1();
    this.mode = 'act';
    this.actSlot = slot;
    this.previousModel = null;
    this.context.setMode('act');
    if (!this.modelStick) {
      this.modelManager.switchTo(target);
      // Act 2 never auto-switches: it would walk away from the model just picked.
      // Act 1 gets back whatever it had when it was left.
      this.modelManager.setAutoSwitch(slot === 1 ? (this.primaryAutoSwitch ?? this.config.autoSwitch) : false);
    }
    this.context.setSystemPrompt(this.modelManager.getCurrentModel());
    return { model: this.modelManager.getCurrentModel() };
  }

  /** Record Act 1's model and auto-switch setting on the way out of it. */
  private leavingAct1(): void {
    if (this.mode !== 'act' || this.actSlot !== 1) return;
    this.primaryModel = this.modelManager.getCurrentModel();
    this.primaryAutoSwitch = this.modelManager.getAutoSwitch();
  }

  /** Shift+Tab: Act 1 -> Act 2 -> Chat -> Act 1. Act 2 is skipped when it has no model. */
  cycleMode(): { mode: AgentMode; slot: ActSlot; model: string } {
    if (this.mode === 'chat') {
      this.setAct(1);
    } else if (this.actSlot === 1 && this.slotModel(2)) {
      this.setAct(2);
    } else {
      this.enterChatMode();
    }
    return { mode: this.mode, slot: this.actSlot, model: this.modelManager.getCurrentModel() };
  }

  /** Enter chat mode — web tools only, fastest conversational model from roster (unless model_stick is on) */
  enterChatMode(): { model: string } {
    this.leavingAct1();
    this.mode = 'chat';
    this.previousModel = this.modelManager.getCurrentModel();
    this.context.setMode('chat');

    if (!this.modelStick) {
      // Use roster's chat model if available
      const chatModel = this.roster?.chat;
      if (chatModel && this.modelManager.getProfile(chatModel)) {
        this.modelManager.switchTo(chatModel);
        this.modelManager.setAutoSwitch(false);
        this.context.setSystemPrompt(chatModel);
        return { model: chatModel };
      }

      // Fallback: pick a fast standard-tier model
      const standardModels = this.modelManager.getModelsByTier('standard')
        .sort((a, b) => b.score - a.score);
      const lightModels = this.modelManager.getModelsByTier('light')
        .filter(m => m.parameterCount >= 3)
        .sort((a, b) => b.score - a.score);

      const best = standardModels[0] || lightModels[0];
      // No chat model known (e.g. lockModel, which discovers nothing else):
      // the primary, not whichever act slot we came from.
      const target = best?.name ?? this.primaryModel ?? this.modelManager.getCurrentModel();
      this.modelManager.switchTo(target);
      this.modelManager.setAutoSwitch(false);
      this.context.setSystemPrompt(target);
    } else {
      this.context.setSystemPrompt(this.modelManager.getCurrentModel());
    }

    return { model: this.modelManager.getCurrentModel() };
  }

  /** Detect image paths in user messages and return base64 data for vision models.
   *  Supports three forms:
   *    1. Absolute / explicit-relative paths (`/tmp/x.png`, `./x.png`, `~/x.png`).
   *    2. `@<path>` mention syntax that survived expandFileMentions
   *       (which now skips images so extractImages handles them).
   *    3. Bare filenames in cwd (`x.png`) when they exist on disk.
   */
  private async extractImages(message: string): Promise<string[]> {
    const ext = '(?:png|jpg|jpeg|gif|webp|bmp)';
    // Either a path with explicit prefix, an @-mention, or a bare filename.
    const pathPattern = new RegExp(
      `(?:^|\\s)(@?(?:(?:/|\\./|~/|[A-Za-z]:\\\\)[\\w./-]+\\.${ext}|[\\w./-]+\\.${ext}))`,
      'gi',
    );
    const matches = [...message.matchAll(pathPattern)];
    if (matches.length === 0) return [];

    const images: string[] = [];
    const seen = new Set<string>(); // dedupe paths mentioned multiple times
    for (const match of matches) {
      let filePath = match[1].trim();
      if (filePath.startsWith('@')) filePath = filePath.slice(1);
      if (filePath.startsWith('~/')) {
        filePath = resolve(process.env.HOME || '~', filePath.slice(2));
      } else {
        filePath = resolve(process.cwd(), filePath);
      }
      if (seen.has(filePath)) continue;
      seen.add(filePath);
      if (existsSync(filePath)) {
        try {
          const data = await readFileAsync(filePath);
          images.push(data.toString('base64'));
        } catch { /* skip unreadable */ }
      }
    }
    return images;
  }

  /** Find a vision-capable model from the roster or model list */
  private findVisionModel(): string | null {
    // Check all models for vision capability
    const visionModels = this.modelManager.getAllModels()
      .filter(m => m.capabilities.includes('vision'))
      .sort((a, b) => b.score - a.score);
    return visionModels[0]?.name || null;
  }

  /** Expand @file mentions in user messages — reads the file and appends content */
  /** Expand @file mentions — searches cwd + additional dirs.
   *  Image extensions are deliberately skipped here; extractImages handles
   *  them downstream as base64 attachments. Inlining a binary file as text
   *  would corrupt the message. */
  private async expandFileMentions(message: string): Promise<string> {
    const mentionPattern = /@([\w./-]+(?:\.\w+))/g;
    const imageExts = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']);
    const mentions = [...message.matchAll(mentionPattern)];
    if (mentions.length === 0) return message;

    const searchDirs = this.context.getSearchDirs();
    const fileContents: string[] = [];
    for (const match of mentions) {
      const ext = match[1].split('.').pop()?.toLowerCase() ?? '';
      if (imageExts.has(ext)) continue; // hand off to extractImages
      // Try each search directory until we find the file
      for (const dir of searchDirs) {
        const filePath = resolve(dir, match[1]);
        if (existsSync(filePath)) {
          try {
            const content = await readFileAsync(filePath, 'utf-8');
            const lines = content.split('\n');
            const preview = lines.length > 200
              ? lines.slice(0, 200).join('\n') + `\n... (${lines.length - 200} more lines)`
              : content;
            fileContents.push(`\n<file path="${relative(process.cwd(), filePath)}">\n${preview}\n</file>`);
            break;
          } catch { /* skip unreadable */ }
        }
      }
    }

    if (fileContents.length === 0) return message;
    return message + '\n\n' + fileContents.join('\n');
  }

  // ─── Plan kept across compaction ─────────────────────────────────
  //
  // In memory, for this session only. It used to be `.veepee/plan.md` in the
  // working tree, re-read from disk after every compaction and injected as
  // "immediately execute the next incomplete step without waiting for user
  // input". So a plan from an EARLIER session — or one committed to the repo,
  // as Nightly Engineer branches did — became an order, and a four-line
  // numbered answer ("2. rm -rf hello.txt", held under verify) was saved as one.
  // Step tracking is todo_write's job now; this only keeps the last plan in view.

  /** Remember the plan if this reply is one. */
  private rememberPlan(content: string): void {
    if (isPlanContent(content)) this.sessionPlan = content;
  }

  /** Load optimal context sizes from latest benchmark results */
  private async loadBenchmarkContextSizes(): Promise<void> {
    const latestPath = resolve(process.env.HOME || '~', '.veepee-code', 'benchmarks', 'latest.json');
    if (!existsSync(latestPath)) return;

    try {
      const data = await readFile(latestPath, 'utf-8');
      const results = JSON.parse(data) as BenchmarkResult[];
      for (const r of results) {
        if (r.context?.optimalSize) {
          this.optimalContextSizes.set(r.model, r.context.optimalSize);
        }
      }
    } catch {
      // Non-critical — use defaults
    }
  }

  /** Get the optimal num_ctx for the current model */
  private getOptimalContext(model: string): number | undefined {
    return this.optimalContextSizes.get(model);
  }

  /**
   * Learn which models the direct server really serves, once.
   *
   * Assuming "the primary" broke any primary the direct server does not have:
   * locking to the AGX's gemma4:26b-a4b in a DGX-direct + gateway setup sent it
   * to the DGX and got `404 The model gemma4:26b-a4b does not exist`, although
   * the gateway serves it. Routing by the server's /v1/models list sends each
   * model where it lives. If the list cannot be fetched, the assumption stands.
   */
  private async ensureDirectModels(): Promise<void> {
    if (!this.openaiBackend || this.directModelsListed || !this.config.openaiBaseUrl) return;
    this.directModelsListed = true;
    if (isDirectOnly(this.config)) return; // no gateway: everything is direct anyway
    try {
      const base = this.config.openaiBaseUrl.replace(/\/+$/, '').replace(/\/v1$/, '');
      const headers: Record<string, string> = this.config.openaiApiKey ? { authorization: `Bearer ${this.config.openaiApiKey}` } : {};
      const res = await fetch(`${base}/v1/models`, { headers, signal: AbortSignal.timeout(5_000) });
      if (!res.ok) return;
      const ids = (((await res.json()) as { data?: Array<{ id?: string }> }).data ?? []).map(m => m.id).filter((x): x is string => !!x);
      if (ids.length > 0) this.directModels = new Set(ids);
    } catch { /* keep the primary assumption */ }
  }

  /** The model to use now: the primary, unless it failed recently, in which
   *  case the first fallback that has not. All down = try the primary again. */
  private pickModel(primary: string): string {
    const now = Date.now();
    for (const m of [primary, ...this.config.fallbackModels]) {
      if ((this.modelDownUntil.get(m) ?? 0) <= now) return m;
    }
    return primary;
  }

  /** Is `model` going to the direct server, with a gateway that could serve it instead? */
  private canRouteDirectToGateway(model: string): boolean {
    return this.openaiBackend && !!this.config.proxyUrl && !isDirectOnly(this.config)
      && this.directModels.has(model) && this.directDownUntil <= Date.now();
  }

  /** Is there a fallback other than `failing` that is not itself marked down? */
  private hasUsableFallback(failing: string): boolean {
    const now = Date.now();
    return [this.modelManager.getCurrentModel(), ...this.config.fallbackModels]
      .some(m => m !== failing && (this.modelDownUntil.get(m) ?? 0) <= now);
  }

  /** Tools this agent handles itself (its own task list); the rest go to the registry. */
  private async runTool(name: string, args: Record<string, unknown>): Promise<ToolResult> {
    if (name !== this.todoTool.name) return this.registry.execute(name, args);
    const parsed = this.todoTool.schema.safeParse(args);
    if (!parsed.success) {
      return { success: false, output: '', error: `Invalid arguments for todo_write: ${parsed.error.issues.map(i => `${i.path.join('.')}: ${i.message}`).join('; ')}` };
    }
    return this.todoTool.execute(parsed.data as Record<string, unknown>);
  }

  /** Fill in an Ollama model's window when no benchmark did (see ollama-context.ts). */
  private async ensureContextSize(model: string): Promise<void> {
    if (!model || this.optimalContextSizes.has(model) || this.contextProbed.has(model)) return;
    this.contextProbed.add(model);
    // A direct OpenAI-compatible server sets its own window, but vcode still
    // has to KNOW it to size output and compaction; it publishes max_model_len.
    const size = this.clientFor(model).isAdapter
      ? await vllmMaxModelLen(this.config.openaiBaseUrl ?? '', model, this.config.openaiApiKey)
      : await ollamaNumCtx(this.config, model);
    if (size) this.optimalContextSizes.set(model, size);
  }

  getContext(): ContextManager {
    return this.context;
  }

  getModelManager(): ModelManager {
    return this.modelManager;
  }

  getPermissions(): PermissionManager {
    return this.permissions;
  }

  getSubAgents(): SubAgentManager {
    return this.subAgents;
  }

  setEffort(level: EffortLevel): void {
    this.effort = level;
  }

  getEffort(): EffortLevel {
    return this.effort;
  }

  /**
   * Get Ollama options based on effort level.
   * Effort controls output length only; sampling temp/top_p/etc come from
   * QWEN_CODING_PRESET (Qwen-recommended values for thinking-mode coding).
   */
  /**
   * The output ceiling, clamped to what the context window can still hold.
   *
   * `num_predict` is not free: the server rejects the whole request when
   * prompt + requested output exceeds the model's window. Raising the ceiling to
   * 16384 this morning fixed truncated tool arguments and immediately created a
   * new failure at the other end — a 52-call session died on
   * `HTTP 400: maximum context length is 131072, you requested 16384 output
   * tokens and your prompt contains …`. The request was refused entirely, so the
   * turn produced nothing at all.
   *
   * A ceiling has to be a budget against the remaining room, not a constant. The
   * floor exists because a request that can only produce 200 tokens is not worth
   * making — better to compact and try again with room to answer.
   */
  private outputBudget(toolSchemaTokens = 0): { num_predict: number } {
    const ceiling = this.getEffortOptions().num_predict;
    const limit = this.context.getContextLimit();
    // The tool definitions are part of the prompt the server counts, and with
    // ~30 tools they are several thousand tokens on their own — far more than a
    // flat reserve covered. gemma4 (32k window) was asked for 15k of output on
    // a prompt vLLM counted ~2k larger than our estimate, every attempt.
    const prompt = this.context.currentPromptTokens() + toolSchemaTokens + this.promptUndercount;
    const RESERVE = 1_024; // chat template overhead and residual estimate error
    const room = limit - prompt - RESERVE;
    if (room >= ceiling) return { num_predict: ceiling };
    return { num_predict: Math.max(1_024, room) };
  }

  private getEffortOptions(): { num_predict: number } {
    // num_predict is a CEILING (reasoning + answer), not a target: a response that
    // finishes naturally costs nothing extra, the cap only prevents truncation on
    // long ones. In act/plan mode thinking is ON, so reasoning (~500-2000 tokens on
    // Qwen3.6) eats into this budget BEFORE the answer — hence the generous headroom.
    // Old values (256/1024/4096) predate thinking-mode and truncated answers; low=256
    // couldn't even fit the reasoning, so it emitted no answer at all.
    // RAISED TWICE on 2026-08-24, the second time because 8192 was still not
    // enough: a real replay task hit FOUR truncated tool calls in one run — a
    // whole `server/lib.js`, a heredoc via bash, then the same file again —
    // because a new source file plus its reasoning does not fit in 8k. The model
    // was not being verbose; the file was simply that size.
    //
    // RAISED 2026-08-24 after measuring what the old ceiling actually cost.
    // At medium (3072) the budget covers reasoning (~500-2000 on Qwen3.6) plus
    // the answer — which means a `write_file` whose content is a real source
    // file cannot fit. The stream is then cut mid-JSON, the arguments arrive
    // unparseable, and the call never runs. Two of six real-repo tasks died
    // exactly that way, both on files of a size an engineer writes without
    // thinking about it. A ceiling costs nothing when a response ends
    // naturally; this one was truncating the work.
    switch (this.effort) {
      case 'low': return { num_predict: 4096 };
      case 'high': return { num_predict: 32768 };
      case 'medium':
      default: return { num_predict: 16384 };
    }
  }

  /** Abort the current running agent loop (called on Ctrl+C) */
  /**
   * Deliver a message to the running agent at the next turn boundary.
   *
   * The reversible rung of the ladder. A guard that has INFERRED something —
   * that the agent looks stuck, that a deadline is approaching — may say so and
   * let the model decide; only a PROVEN state (an exit code, a byte count, a
   * wall clock) earns a terminal action. Every guard that killed a working run
   * tonight did it by inferring a state and acting as though it knew.
   */
  notify(text: string): void {
    this.pendingNotices.push(text);
  }

  private pendingNotices: string[] = [];

  abort(): void {
    this.abortController?.abort();
    // Interrupt must reach the WORK, not just the model stream. Aborting the
    // stream while `npm test` keeps running holds the terminal for the tool's
    // whole timeout with the Ctrl+C visibly ignored — the harness looks wedged
    // at exactly the moment the user is trying to take control back.
    killRunningBashCommands();
  }

  isRunning(): boolean {
    return this.runLock;
  }

  /**
   * Claim the agent for a single run, then delegate to the real loop.
   *
   * The persistent server shares ONE Agent across /v1/chat/completions,
   * /rc/send and the TUI. Without this guard two overlapping requests both
   * append to the same `context` (interleaving two conversations into one
   * history) and the second clobbers `this.abortController`, so the first run
   * becomes unabortable and whichever finishes first nulls the handle for both.
   *
   * This is deliberately a plain method that *returns* a generator rather than
   * being `async *` itself: the check-and-set below runs synchronously at call
   * time, so there is no await gap for a second caller to slip through. An
   * async generator's body would not execute until the first `next()`, which
   * would leave exactly that race open.
   */
  run(
    userMessage: string,
    options?: {
      permissionMode?: PermissionMode;
      allowedTools?: string[] | null;
      onTurnBoundary?: () => string[] | Promise<string[]>;
    },
  ): AsyncGenerator<AgentEvent> {
    if (this.runLock) throw new AgentBusyError();
    this.runLock = true;
    this.registry?.beginRun();
    this.checkpointedThisTurn = false;
    return this.withRunLock(this._run(userMessage, options));
  }

  /** Release the run lock once the inner loop finishes, throws, or is abandoned. */
  private async *withRunLock(inner: AsyncGenerator<AgentEvent>): AsyncGenerator<AgentEvent> {
    try {
      yield* inner;
    } finally {
      // Policy workers cannot advertise idle while abandoned tool work still runs.
      if (this.registry?.policy) await this.registry.settle();
      this.runLock = false;
      // The turn is over — however it ended. This sits in the SAME finally as the
      // run lock on purpose: a run that threw, was aborted, or was abandoned
      // mid-stream must not leave vcode advertising `working` forever, which would
      // make every supervisor watching it wait on a turn that is already gone.
      reportAgentState('idle');
    }
  }

  /**
   * Pick the transport for a given model.
   *
   * With `llmBackend: "openai"` the agent talks to ONE vLLM server, which
   * serves exactly one model — the DGX serves Qwen3.6, the AGX serves Gemma 4,
   * and they don't switch. So a turn that names a different model (`/review`
   * routing through `reviewModel`, say) cannot go to the direct endpoint: it
   * would come back as a model-not-found. Those turns go through the gateway,
   * which fronts the whole fleet and can reach whichever box holds that model.
   *
   * Falling back to the gateway is never *wrong*, only an extra hop — so an
   * unknown model routes there rather than failing.
   *
   * `isAdapter` tells the caller whether the chosen client is OpenAIChatClient:
   * only it consumes `signal`, and passing that to the Ollama client would
   * serialize an AbortSignal into the request body.
   */
  private clientFor(model: string): { client: Ollama; isAdapter: boolean } {
    // With no gateway there is nowhere else to go: every model is direct.
    // Once the direct server's connection has failed, the gateway (which routes
    // around a dead box) serves everything until it is tried again.
    const directDown = this.directDownUntil > Date.now() && !!this.config.proxyUrl;
    if (!this.openaiBackend || isDirectOnly(this.config) || (this.directModels.has(model) && !directDown)) {
      return { client: this.ollama, isAdapter: this.openaiBackend };
    }
    if (!this.gatewayClient) {
      this.gatewayClient = new Ollama({
        host: this.config.proxyUrl,
        headers: { 'x-ollama-source': 'vcode' },
      });
    }
    return { client: this.gatewayClient, isAdapter: false };
  }

  /** Attach file checkpointing. Optional — without it the agent behaves exactly
   *  as before, just with no rewind. */
  setCheckpointManager(cm: CheckpointManager | null): void {
    this.checkpoints = cm;
  }

  /**
   * Snapshot the working tree before the first tool of a turn runs.
   *
   * Taken lazily rather than at turn start so read-only turns cost nothing, and
   * before the tool rather than after so the checkpoint represents "the state
   * you can get back to". Deliberately swallows everything: losing a checkpoint
   * is an inconvenience, failing the user's turn because of one is not.
   */
  private async checkpointOnce(userMessage: string): Promise<void> {
    if (!this.checkpoints || this.checkpointedThisTurn) return;
    this.checkpointedThisTurn = true;
    try {
      await this.checkpoints.snapshot(userMessage);
    } catch { /* best effort */ }
  }

  /**
   * Look for enumerated sets that fell out of step with the files just edited.
   *
   * Scope is deliberately narrow: only files in the same directories as the
   * ones edited this turn, and only source files. A repo-wide scan would be
   * slow and would surface unrelated lists — this is a nudge, and a noisy nudge
   * costs a turn every time it is wrong.
   */
  private async buildCompletenessNudgeForTurn(editedPaths: Set<string>): Promise<string | null> {
    try {
      const { readdir, readFile: rf } = await import('node:fs/promises');
      const { dirname, join: joinPath, extname, relative: rel } = await import('node:path');
      const SOURCE = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.py', '.go', '.rs']);
      const MAX_FILES = 40;
      const MAX_BYTES = 200_000;

      const dirs = new Set([...editedPaths].map((p) => dirname(p)));
      const files = new Map<string, string>();
      for (const dir of dirs) {
        let entries: string[] = [];
        try { entries = await readdir(dir); } catch { continue; }
        for (const name of entries) {
          if (files.size >= MAX_FILES) break;
          if (!SOURCE.has(extname(name))) continue;
          const full = joinPath(dir, name);
          try {
            const content = await rf(full, 'utf-8');
            if (content.length <= MAX_BYTES) files.set(rel(process.cwd(), full), content);
          } catch { /* unreadable — skip */ }
        }
      }
      if (files.size < 2) return null;

      const { findExtensionGaps, buildCompletenessNudge, selectGaps } = await import('./completeness.js');
      // Gaps in files the model EDITED are kept — see `selectGaps`. Excluding
      // them threw away the strongest case there is: a file where the model
      // added the new member in one place and missed the enumeration two
      // functions below.
      const editedRel = new Set([...editedPaths].map((p) => rel(process.cwd(), p)));
      const gaps = selectGaps(findExtensionGaps(files), editedRel, files);
      return buildCompletenessNudge(gaps);
    } catch {
      return null; // never fail a turn over a nudge
    }
  }

  setModel(model: string): void {
    this.modelManager.switchTo(model);
    this.context.setSystemPrompt(model);
  }

  /** Run the agent loop for a user message, yielding events as they occur */
  private async *_run(
    userMessage: string,
    options?: {
      permissionMode?: PermissionMode;
      allowedTools?: string[] | null;
      /** Called between tool batches and the next LLM call. Returned strings
       *  are added to context as user messages, in order, before the next
       *  `ollama.chat()` invocation. Used to deliver "steering" messages
       *  the user submitted mid-turn without aborting the run. */
      onTurnBoundary?: () => string[] | Promise<string[]>;
    },
  ): AsyncGenerator<AgentEvent> {
    this._runResult = null;
    const permissionMode = options?.permissionMode || 'interactive';
    const allowedTools = this.registry.policy
      ? new Set(this.registry.policy.intersect(options?.allowedTools))
      : options?.allowedTools ? new Set(options.allowedTools) : null;
    const onTurnBoundary = options?.onTurnBoundary;
    this.abortController = new AbortController();

    // A turn is now in flight. Reported to veeWM (and written to the terminal
    // title) so anything watching this window can distinguish "busy" from the two
    // states that look identical from outside: idle, and stopped for approval.
    reportAgentState('working');

    // UserPromptSubmit hook — fires on raw user input, before any expansion
    // or model interaction. Hook stdout is shown to the user; non-zero exit
    // does NOT block the run (advisory only). Lets users automate things
    // like "log every prompt" or "warn if prompt mentions production".
    yield* this._fireHooks('UserPromptSubmit', { prompt: userMessage, cwd: process.cwd() });

    // Expand @file mentions — read files and append content
    const expandedMessage = this.registry.policy ? userMessage : await this.expandFileMentions(userMessage);

    // Detect images in message and switch to vision model if needed
    const images = this.registry.policy ? [] : await this.extractImages(expandedMessage);
    let visionModelSwitch: string | null = null;
    if (images.length > 0) {
      const visionModel = this.findVisionModel();
      if (visionModel && visionModel !== this.modelManager.getCurrentModel()) {
        visionModelSwitch = this.modelManager.getCurrentModel(); // save to restore later
        this.modelManager.switchTo(visionModel);
        this.context.setSystemPrompt(visionModel);
        yield { type: 'model_switch', content: 'Switching to vision model for image analysis', from: visionModelSwitch, to: visionModel };
      } else if (!visionModel) {
        yield { type: 'info', content: 'No vision model available — image will be described by path only' };
      }
    }

    // SessionStart: once, before the first turn. Its output is context for the
    // model (Claude Code's contract), e.g. memory recalled for this project.
    if (!this.sessionStarted) {
      this.sessionStarted = true;
      const started = this.registry.policy ? [] : await runHooks('SessionStart', { cwd: process.cwd() } as never);
      const context = started.filter(r => r.exitCode === 0 && r.stdout.trim()).map(r => r.stdout.trim()).join('\n\n');
      if (context) {
        this.context.addUser(`[SYSTEM] Session context from SessionStart hooks:\n${context}`);
        yield { type: 'info', content: `SessionStart hooks added ${context.length} characters of context` };
      }
    }

    this.context.addUser(expandedMessage);

    // Set context limit from benchmarks or model metadata
    await this.ensureDirectModels();
    await this.ensureContextSize(this.modelManager.getCurrentModel());
    const ctxLimit = this.getOptimalContext(this.modelManager.getCurrentModel());
    if (ctxLimit) {
      this.context.setContextLimit(ctxLimit);
    }

    // Pre-compaction snapshot at 90%: with no plan written, keep the recent
    // assistant turns and knowledge state instead (in memory; costs no tokens).
    if (this.context.isContextCritical() && !this.sessionPlan) {
      const recentAssistant = this.context.getAllMessages()
        .filter(m => m.role === 'assistant' && m.content)
        .slice(-3)
        .map(m => m.content)
        .join('\n\n---\n\n');
      if (recentAssistant.length > 100) {
        const ks = this.context.getKnowledgeState().serialize();
        this.sessionPlan = `## Knowledge State\n\n${ks}\n\n## Recent Context\n\n${recentAssistant}`;
      }
    }

    // Check for context compaction
    if (this.context.needsCompaction()) {
      // Prune before summarizing. Truncating stale tool output is LOSSLESS in
      // structure — the call and its arguments survive, only the bulk goes — and
      // costs no model call, no latency and no chance of the summarizer dropping
      // or inventing a fact. Reach for the summarizer only if this wasn't enough.
      const reclaimed = this.context.pruneToolOutputs();
      if (reclaimed > 0) {
        yield { type: 'info', content: `Pruned ~${reclaimed.toLocaleString()} tokens of stale tool output from history — more of the conversation now fits in the window` };
        if (!this.context.needsCompaction()) {
          this.context.getKnowledgeState().save().catch((err) => {
          // Losing the knowledge state silently means the next session starts
          // with less than it should and nobody knows why. It must not fail the
          // turn — but it must not be invisible either.
          process.stderr.write(`[knowledge-state] save failed: ${err instanceof Error ? err.message : String(err)}\n`);
        });
          return;
        }
      }
      const retryEvents: Array<{ attempt: number; projected: number; limit: number }> = [];
      yield* this._fireHooks('PreCompact', { cwd: process.cwd(), messageCount: this.context.messageCount() });
      const compacted = await this.context.compactWithRetry(
        createChatClient(this.config),
        this.modelManager.getCurrentModel(),
        this.config.summarizerModel,
        {
          onRetry: (attempt, projected, limit) => {
            retryEvents.push({ attempt, projected, limit });
          },
        },
      );
      if (compacted) {
        yield { type: 'info', content: 'Compacted conversation to free context space' };
        for (const r of retryEvents) {
          yield { type: 'info', content: `Compacting harder (attempt ${r.attempt}) — projected ${r.projected} > ${Math.round(r.limit * 0.85)} cutoff` };
        }

        // Recover saved plan after compaction so the model doesn't lose it
        if (this.sessionPlan) {
          this.context.addUser(RESTORED_PLAN_NOTE + this.sessionPlan);
          yield { type: 'info', content: "Restored this session's plan after compaction" };
        }
      }
    }

    // Stuck loop detection: hash-signature window (input + output).
    // Catches ABAB oscillation, not just N consecutive identical calls.
    const recentSteps: SignedStep2[] = [];
    const MAX_TURNS_WITHOUT_OUTPUT = 15;
    let turnsWithoutUserContent = 0;
    // Tier 3 #1: force-act guard — track whether the model has taken ANY action this
    // message, and whether we've already nudged it to stop narrating and act.
    let hasActedThisMessage = false;
    const announcedFallback = new Set<string>();
    let contextWindowRetries = 0;
    this.todoNudged = false; // one task-list reminder per user message
    let forcedActCount = 0;
    // Daily-driver #1 (self-repair): true once code is edited, cleared when the model runs
    // something (bash) — so we can force a verify-and-fix turn if it finishes without running it.
    let codeChangedUnverified = false;
    let forcedCompletenessOnce = false;
    let repeatedFailureWarned = false;
    /** True once we've warned the model it is repeating streamed text. */
    let contentRepetitionWarned = false;
    /** Files this run wrote to, for the incomplete-extension check. */
    const editedPaths = new Set<string>();
    let forcedVerifyOnce = false;

    // A hard ceiling on one user message. The loop's only other exits are the
    // model choosing to stop, a stuck-signature match (which needs SIX
    // byte-identical name+args+result tuples, so an alternating or
    // output-varying loop never trips it), 15 turns with no visible content
    // (reset by a single word of output), an error, or an abort. None of those
    // bound a model that is making slow, plausible, unproductive progress.
    try {
      for (let turn = 0; ; turn++) {
      // Drain anything queued for the agent since the last turn.
      while (this.pendingNotices.length > 0) {
        const notice = this.pendingNotices.shift()!;
        this.context.addUser(notice);
        yield { type: 'info', content: notice.slice(0, 120) };
      }

      const turnLimit = this.registry.policy?.maxTurns ?? MAX_TURNS_PER_MESSAGE;
      if (turn >= turnLimit) {
        yield { type: 'error', error: `Stopped: ${turnLimit} turns on one message without finishing. The work so far is kept — send a narrower instruction to continue.` };
        this.abortController = null;
        return;
      }
      // Check if model should switch (only after the first turn of a message)
      if (turn > 0) {
        const signals = this.context.getSignals();
        const newModel = this.modelManager.evaluate(signals);
        if (newModel) {
          this.context.setSystemPrompt(newModel);
          yield { type: 'model_switch', from: this.modelManager.getCurrentModel(), to: newModel };
        }

        // Steering boundary: drain any messages the user submitted mid-turn
        // and inject them as user messages before the next LLM call. Runs
        // AFTER tool results have been added to context, BEFORE the next
        // ollama.chat. Pre-empts whatever the model would have done next.
        if (onTurnBoundary) {
          try {
            const steering = await onTurnBoundary();
            for (const msg of steering) {
              const trimmed = msg.trim();
              if (!trimmed) continue;
              this.context.addUser(`[USER STEERING] ${trimmed}\n\n(The user changed direction mid-turn. Re-evaluate based on this new input before continuing.)`);
              yield { type: 'info', content: `Steering: ${trimmed.slice(0, 80)}${trimmed.length > 80 ? '…' : ''}` };
            }
          } catch {
            // Steering callback failures are non-fatal — keep running.
          }
        }
      }

      const primaryModel = this.modelManager.getCurrentModel();
      const currentModel = this.pickModel(primaryModel);
      if (currentModel !== primaryModel && !announcedFallback.has(currentModel)) {
        announcedFallback.add(currentModel);
        yield { type: 'info', content: `${primaryModel} is unreachable — continuing on ${currentModel}` };
      }

      // Build messages with system prompt
      const contextMessages = this.context.getMessages();
      const messages: Message[] = [
        { role: 'system', content: this.context.getSystemPrompt() },
        ...contextMessages,
      ];

      // Inject images into the last user message if present
      if (images.length > 0 && messages.length > 0) {
        const lastMsg = messages[messages.length - 1];
        if (lastMsg.role === 'user') {
          (lastMsg as unknown as { images: string[] }).images = images;
        }
      }

      // Stall timeout: 5 minutes with no chunks = assume Ollama is hung
      // Resets on each chunk, so model loading time doesn't trigger it
      const STALL_TIMEOUT_MS = 5 * 60 * 1000;
      let stallTimer: ReturnType<typeof setTimeout> | null = null;
      const resetStallTimer = () => {
        if (stallTimer) clearTimeout(stallTimer);
        stallTimer = setTimeout(() => {
          this.abortController?.abort();
        }, STALL_TIMEOUT_MS);
      };

      // Stream LLM response with thinking detection
      let fullContent = '';
      /** Reasoning delivered on its own channel (vLLM `reasoning` / gateway
       *  `thinking`). Kept OUT of `fullContent` — it is not the assistant's
       *  answer and must not reach the context or the act/verify heuristics —
       *  but retained so a turn that spent its whole budget reasoning can still
       *  surface something instead of ending silent. */
      let fullThinking = '';
      let toolCalls: ToolCall[] = [];
      // Names of the tools offered this turn (chat mode offers a subset).
      let offeredTools = new Set<string>();
      // Holds back text that may be a tool call written as text (see gate).
      let gate = new TextToolCallGate(false);
      let inThinking = false;
      let thinkingBuffer = '';
      let evalCount = 0;
      let promptEvalCount = 0;
      /** How much of this prompt the server served from its KV prefix cache.
       *  The single most useful number for turn latency, and vcode was blind to
       *  it: a prompt whose last 12 characters change every turn re-prefills the
       *  entire conversation, and nothing in the harness could see that. */
      let promptCachedCount = 0;
      let evalDuration = 0;
      const generationStarted = Date.now();
      let firstEventAt: number | null = null;
      this.trace({ kind: 'model_turn_started', turn: turn + 1 });

      try {
        // Use optimal context size from benchmarks if available
        await this.ensureContextSize(currentModel); // the model can change mid-turn
        const numCtx = this.getOptimalContext(currentModel);
        // …and so can its window (a fallback to a smaller model).
        if (numCtx && numCtx !== this.context.getContextLimit()) this.context.setContextLimit(numCtx);

        // Mode-specific settings:
        // verify (any mode): mutating tools refused by permissions until request_approval
        // act:  thinking ON (Qwen3.6 needs CoT for reliable tool use — without
        //        it, the model produces "I can't SSH from this environment"
        //        fluff and skips bash calls entirely), all tools, auto-switch
        // chat: thinking OFF (proxy translates to enable_thinking=false on
        //        Qwen3 vLLM since 2026-05-04), web/search tools only
        // The act-mode flip from OFF to ON corrects a regression introduced
        // when the proxy started actually honoring `think:false`. Previously
        // act sent think:false but the proxy silently dropped it, so the
        // model thought anyway. Once the proxy began translating it, act
        // mode genuinely went to instruct mode and tool-use quality cratered.
        const useThinking = this.mode !== 'chat';
        // Plan mode gets the SAME tools as act. The only difference between the
        // two is which model answers.
        //
        // It used to filter out bash/edit_file/write_file/multi_edit as a hard
        // gate. That produced the worst possible failure: the model could not
        // see the tools, so it could not tell the user they were unavailable or
        // ask for them — it silently improvised. Asked to analyse config drift,
        // it found the project's own pinky_drift.py, read it, and rebuilt its
        // output with ~50 read-only calls because bash was not in its list.
        //
        // Permissions are the right layer for "do not let it mutate things",
        // and they already prompt per call. A mode is a poor access-control
        // mechanism: it is invisible to the thing being controlled.
        let tools = this.mode === 'chat'
          ? this.registry.toOllamaTools().filter(t => {
              const name = t.function?.name || '';
              return CHAT_TOOLS.includes(name);
            })
          : this.registry.toOllamaTools();
        // request_approval only means something in verify; elsewhere it is schema tokens for nothing.
        if (!this.verify) tools = tools.filter(t => t.function?.name !== 'request_approval');

        // Filter tools for API requests with client-constrained tool sets
        if (allowedTools) {
          tools = tools.filter(t => allowedTools.has(t.function?.name || ''));
        }
        offeredTools = new Set(tools.map(t => t.function?.name || ''));
        gate = new TextToolCallGate(offeredTools.size > 0);
        // ~3 characters per token for JSON schemas (dense punctuation and names).
        const toolSchemaTokens = tools.length > 0 ? Math.ceil(JSON.stringify(tools).length / 3) : 0;
        const effortOpts = this.outputBudget(toolSchemaTokens);
        // Sampling preset: chat mode → conversational/general; act/plan → coding.
        // Both Qwen-recommended; harmless on other Qwen3.x models, only wrong if
        // the user unlocks to a non-Qwen family (no current path does this).
        // Chat mode: thinking is actually disabled (proxy translates think:false
        // → chat_template_kwargs.enable_thinking=false for Qwen3 on vLLM), so we
        // use Qwen's Instruct preset. Act/plan keep thinking + Coding preset.
        const samplingPreset = this.mode === 'chat' ? QWEN_INSTRUCT_PRESET : QWEN_CODING_PRESET;

        // Route to whichever endpoint serves this model — the direct vLLM
        // server only has the primary; anything else goes via the gateway.
        const { client: chatClient, isAdapter } = this.clientFor(currentModel);

        const chatRequest = () => ({
          model: currentModel,
          messages,
          ...(tools.length > 0 ? { tools } : {}),
          stream: true as const,
          think: useThinking && !this.noThinkModels.has(currentModel),
          keep_alive: '30m',
          // Only the openai adapter consumes `signal`; never send it to the
          // Ollama client (it would serialize into the request body).
          ...(isAdapter && this.abortController ? { signal: this.abortController.signal } : {}),
          options: {
            ...samplingPreset,
            ...(numCtx ? { num_ctx: numCtx } : {}),
            ...effortOpts,
          },
        });

        // Retry wrapper. Was: ONE attempt, fixed 3s, four connection-error
        // strings — which covers a server that is down and nothing else. It did
        // not cover what this fleet actually produces: the DGX engine wedging
        // and the watchdog taking ~63s to restore it, a 429 from a busy gateway,
        // or a 503 while a model loads. A turn dying inside that window loses
        // the whole run. See `retry.ts` for why context-overflow is excluded and
        // why `Retry-After` wins over any backoff we would compute.
        const chatWithRetry = async () => {
          for (let attempt = 1; ; attempt++) {
            try {
              return await chatClient.chat(chatRequest() as never);
            } catch (err) {
              // A user interrupt is not a transport fault; never retry it.
              if (this.abortController?.signal.aborted) throw err;
              if (!this.noThinkModels.has(currentModel) && /does not support thinking/i.test(String((err as Error)?.message ?? err))) {
                this.noThinkModels.add(currentModel);
                continue; // same request without `think` (chatRequest reads the set)
              }
              // With a fallback available, stop waiting on a server that cannot
              // be reached after two attempts (~1 min) instead of up to ten
              // minutes; the turn continues on the next model.
              if (attempt >= FALLBACK_AFTER_ATTEMPTS && isTransportFailure(err) && (this.canRouteDirectToGateway(currentModel) || this.hasUsableFallback(currentModel))) throw err;
              const decision = retryDecision(err, attempt);
              if (!decision.retry) throw err;
              await new Promise(r => setTimeout(r, decision.delayMs));
              // The abort signal is rebuilt into every attempt by chatRequest():
              // a retried stream that cannot be aborted is the orphaned stream
              // that wedges vLLM, which is what the signal was added for.
            }
          }
        };

        resetStallTimer();
        // One generation per model at a time, across this agent and every
        // subagent it spawns. The slot is held for the whole stream and
        // released before tools run, so it never blocks the work between turns.
        const stream = generationLimiter.stream(currentModel, chatWithRetry);

        for await (const chunk of stream) {
          resetStallTimer();

          // Check for abort
          if (this.abortController?.signal.aborted) {
            if (stallTimer) clearTimeout(stallTimer);
            const held = gate.end();
            if (held) yield { type: 'text', content: held };
            yield { type: 'error', error: 'Interrupted by user' };
            this.abortController = null;
            return;
          }

          if (firstEventAt === null) { firstEventAt = Date.now(); this.trace({ kind: 'model_first_event', turn: turn + 1, first_event_ms: firstEventAt - generationStarted }); }
          // Gateway-split reasoning arrives in the Ollama `thinking` field: the DGX
          // runs reasoning-parser OFF, so llm-gateway splits `[trace]</think>[answer]`
          // and emits the trace here. Render it collapsed instead of leaking inline.
          const think = (chunk.message as { thinking?: string }).thinking;
          if (think) {
            fullThinking += think;
            yield { type: 'thinking', content: think };
          }

          if (chunk.message.content) {
            const text = chunk.message.content;
            fullContent += text;

            // Degenerate content repetition: same phrase repeated many times.
            if (fullContent.length >= 500 && !contentRepetitionWarned) {
              const rep = detectContentRepetition(fullContent);
              if (rep) {
                contentRepetitionWarned = true;
                this.notify(
                  `[SYSTEM] You are repeating the same phrase "${rep.repeated.trim()}" ${rep.count} times in a row. `
                  + `Stop repeating and write what you have.`,
                );
                yield { type: 'info', content: 'Warned: degenerate content repetition' };
                continue;
              }
            }

            // Detect <think> tags (used by Qwen, DeepSeek, etc.)
            if (!inThinking && text.includes('<think>')) {
              inThinking = true;
              // Extract any text before <think> tag
              const before = text.split('<think>')[0];
              if (before) { const shown = gate.push(before); if (shown) yield { type: 'text', content: shown }; }
              // Start thinking buffer
              thinkingBuffer = text.split('<think>').slice(1).join('<think>');
              yield { type: 'thinking', content: '...' }; // signal thinking started
              continue;
            }

            // Orphan </think>: reasoning models like Qwen3.6 (served by vLLM
            // without a reasoning parser) emit the thinking trace directly
            // into content and close it with a bare </think> before the final
            // answer. Reclassify everything streamed so far as thinking and
            // reset the TUI's stream buffer so the user only sees the answer.
            if (!inThinking && text.includes('</think>')) {
              const parts = text.split('</think>');
              const beforeClose = parts[0];
              const afterClose = parts.slice(1).join('</think>');
              // Everything streamed before this chunk, plus the portion of
              // this chunk up to the orphan close, was reasoning.
              const streamedBefore = fullContent.slice(0, fullContent.length - text.length);
              const reasoningText = (streamedBefore + beforeClose).trim();

              yield { type: 'reset_stream' };
              gate.reset();
              if (reasoningText) yield { type: 'thinking', content: reasoningText };
              if (afterClose) { const shown = gate.push(afterClose); if (shown) yield { type: 'text', content: shown }; }
              continue;
            }

            if (inThinking) {
              if (text.includes('</think>')) {
                // End of thinking block
                const parts = text.split('</think>');
                thinkingBuffer += parts[0];
                inThinking = false;

                // Yield the full thinking content (collapsed in TUI)
                yield { type: 'thinking', content: thinkingBuffer.trim() };
                thinkingBuffer = '';

                // Any text after </think> is regular output
                const after = parts.slice(1).join('</think>');
                if (after) { const shown = gate.push(after); if (shown) yield { type: 'text', content: shown }; }
              } else {
                thinkingBuffer += text;
                // Periodically update thinking indicator
                if (thinkingBuffer.length % 200 < text.length) {
                  yield { type: 'thinking', content: '...' };
                }
              }
              continue;
            }

            const shown = gate.push(text);
            if (shown) yield { type: 'text', content: shown };
          }

          if (chunk.message.tool_calls && chunk.message.tool_calls.length > 0) {
            toolCalls = chunk.message.tool_calls;
          }

          // Capture eval metrics from final chunk
          const c = chunk as unknown as Record<string, number>;
          if (c.eval_count) evalCount += c.eval_count;
          if (c.prompt_eval_count) promptEvalCount += c.prompt_eval_count;
          if (typeof c.prompt_cached_count === 'number') promptCachedCount = c.prompt_cached_count;
          if (c.eval_duration) evalDuration += c.eval_duration;
        }

        if (stallTimer) clearTimeout(stallTimer);

        // If thinking was still open (malformed output), flush it
        if (inThinking && thinkingBuffer) {
          yield { type: 'thinking', content: thinkingBuffer.trim() };
        }
        // This model completed a reply: it did (some of) the work.
        this.modelsUsed.add(currentModel);
        this.trace({ kind: 'model_turn_finished', turn: turn + 1, duration_ms: Date.now() - generationStarted, output_tokens: evalCount, prompt_tokens: promptEvalCount });
      } catch (err) {
        if (stallTimer) clearTimeout(stallTimer);
        // Whatever the gate was holding is part of the answer; do not lose it.
        const heldOnError = gate.end();
        if (heldOnError) yield { type: 'text', content: heldOnError };
        this.trace({ kind: 'model_turn_interrupted', turn: turn + 1, duration_ms: Date.now() - generationStarted });
        const wasAborted = this.abortController?.signal.aborted;
        this.abortController = null;
        if (wasAborted) {
          yield { type: 'error', error: 'Response timed out or interrupted' };
          return;
        }
        // The server told us its real window ("maximum context length is N") —
        // we had assumed a bigger one. Adopt it, compact if the conversation no
        // longer fits, and redo the step once instead of failing the run.
        const errText = describeError(err);
        const windowMatch = /maximum context length is (\d+)/i.exec(errText);
        if (windowMatch && contextWindowRetries < 2) {
          contextWindowRetries++;
          const window = Number(windowMatch[1]);
          // Did we already know this window? Then resizing changes nothing: our size
          // estimate was wrong, and vLLM's "at least N" is a floor (it stops counting
          // at window - requested + 1), so the real overshoot is unknown. Compact
          // rather than retry the same request with a smaller output ceiling.
          const knewWindow = this.context.getContextLimit() === window;
          this.optimalContextSizes.set(currentModel, window);
          this.context.setContextLimit(window);
          // vLLM also states the prompt's real size ("prompt contains at least N
          // input tokens"). Learn how far our estimate was under, so this retry
          // and every later request leave the right amount of room.
          const counted = /prompt contains at least (\d+) input tokens/i.exec(errText);
          const requested = /requested (\d+) output tokens/i.exec(errText);
          if (counted && requested) {
            const ourEstimate = window - Number(requested[1]) - 1_024;
            const under = Number(counted[1]) - ourEstimate;
            if (under > 0) this.promptUndercount += under + 256;
          }
          yield { type: 'reset_stream' };
          yield { type: 'info', content: `${currentModel} has a ${window}-token window; resizing the request and retrying` };
          if (knewWindow || this.context.projectedTokens() > window * 0.85) {
            yield* this._fireHooks('PreCompact', { cwd: process.cwd(), messageCount: this.context.messageCount() });
            await this.context.compactWithRetry(createChatClient(this.config), currentModel, this.config.summarizerModel);
          }
          continue;
        }
        // The server could not be reached, or dropped the connection mid-reply
        // (a Jetson resetting under load did exactly this and killed the run).
        // Nothing from this attempt reached the context — tool calls only run
        // after the stream — so redo the step on the next model.
        if (isTransportFailure(err) && this.canRouteDirectToGateway(currentModel)) {
          // The direct server died; the gateway may still serve this model elsewhere.
          this.directDownUntil = Date.now() + MODEL_DOWN_MS;
          const note = `direct server ${this.config.openaiBaseUrl} failed (${describeError(err).slice(0, 100)}); ${currentModel} retried via the gateway`;
          this.fallbackNotes.push(note);
          yield { type: 'reset_stream' };
          yield { type: 'info', content: note };
          continue;
        }
        if (isTransportFailure(err) && this.hasUsableFallback(currentModel)) {
          this.modelDownUntil.set(currentModel, Date.now() + MODEL_DOWN_MS);
          const note = `${currentModel} failed (${describeError(err).slice(0, 100)}); continued on a fallback model`;
          this.fallbackNotes.push(note);
          yield { type: 'reset_stream' };
          yield { type: 'info', content: `${currentModel} failed (${describeError(err).slice(0, 120)}) — retrying this step on the next fallback model` };
          continue;
        }
        // Defense: the Ollama SDK's ResponseError class stringifies its
        // `message` arg via the Error constructor — so when vLLM returns an
        // error JSON whose `.error` field is an object (not a string), the
        // SDK ends up with `responseError.message = "[object Object]"`.
        // The original object IS preserved on `responseError.error` though.
        // Same pattern for any custom Error subclass that wraps structured
        // data: try the .error / .body / .response fields before giving up.
        const safeStringify = (v: unknown): string => {
          try {
            const seen = new WeakSet();
            return JSON.stringify(v, (_k, val) => {
              if (typeof val === 'object' && val !== null) {
                if (seen.has(val)) return '[circular]';
                seen.add(val);
              }
              return val;
            }) || String(v);
          } catch {
            return String(v);
          }
        };
        let msg: string;
        if (err instanceof Error) {
          // If the message got corrupted to "[object Object]" or is empty,
          // unwrap any structured data the SDK preserved on the Error.
          const baseMsg = err.message || err.toString();
          const errObj = err as Error & { error?: unknown; body?: unknown; response?: unknown; status_code?: unknown; cause?: unknown };
          const recoverable = errObj.error ?? errObj.body ?? errObj.response ?? errObj.cause;
          if (baseMsg === '[object Object]' && recoverable !== undefined) {
            msg = typeof recoverable === 'string' ? recoverable : safeStringify(recoverable);
          } else if (recoverable !== undefined && typeof recoverable === 'object') {
            // Append structured data when present so users see status codes etc.
            msg = `${baseMsg} ${safeStringify(recoverable)}`;
          } else {
            msg = baseMsg;
          }
        } else if (typeof err === 'string') {
          msg = err;
        } else if (err && typeof err === 'object') {
          msg = safeStringify(err);
        } else {
          msg = String(err);
        }
        yield { type: 'error', error: msg };
        this.context.addAssistant(`Error communicating with model: ${msg}`);
        return;
      }

      // Record actual token usage for context-aware window sizing
      if (promptEvalCount > 0) {
        this.context.recordPromptTokens(promptEvalCount);
      }

      // A turn that produced reasoning, no content and no tool calls has an
      // answer only on the reasoning channel — surface it rather than ending
      // the turn silent. This is the `07727dc` failure (a whole run emitting
      // ONE BYTE) and it is the reason reasoning was ever folded into content;
      // the promotion happens HERE, at the one point where the turn is known to
      // be otherwise empty, instead of for every token of every turn.
      if (!fullContent.trim() && toolCalls.length === 0 && fullThinking.trim()) {
        fullContent = fullThinking.trim();
        yield { type: 'text', content: fullContent };
      }

      // A tool call written as text (small local models do this) becomes a
      // real one — but only if the whole answer is calls to tools offered this
      // turn. See text-tool-calls.ts for why it is that strict.
      const heldText = gate.end();
      const recovered = toolCalls.length === 0 && fullContent.trim()
        ? parseTextToolCalls(answerText(fullContent), (n) => offeredTools.has(n))
        : null;
      if (recovered) {
        // The held text WAS the call: it is run, never shown.
        toolCalls = recovered;
        fullContent = '';
      } else if (heldText) {
        // It only looked like one: show it now, unchanged.
        yield { type: 'text', content: heldText };
      }

      // Add assistant message to context
      this.context.addAssistant(fullContent, toolCalls.length > 0 ? toolCalls : undefined);

      // If no tool calls, the turn is complete
      if (toolCalls.length === 0) {
        // Tier 3 #1: in ACT mode, if the model narrated without ever calling a tool, it
        // analyzed instead of acting (the no-<think> budget leak). Force one more turn to
        // act — once — rather than "completing" with nothing done.
        if (shouldForceAct({ mode: this.mode, hasActedThisMessage,
          alreadyForced: forcedActCount >= FORCE_ACT_MAX_NUDGES, content: fullContent, userMessage })) {
          forcedActCount++;
          this.context.addUser(forceActNudge(fullContent));
          yield { type: 'info', content: 'Nudged: act instead of narrate' };
          continue;
        }
        // Ending with task-list items still open: remind once. Small models
        // declare victory with steps undone; the list says so precisely.
        const openTodos = this.todos.open();
        if (this.mode === 'act' && openTodos.length > 0 && !this.todoNudged) {
          this.todoNudged = true;
          this.context.addUser(`[SYSTEM] Your task list still has ${openTodos.length} open item${openTodos.length === 1 ? '' : 's'}:\n${openTodos.map(t => `- ${t.content}`).join('\n')}\nFinish them, or if one is no longer needed or already done, update the list with todo_write. Then reply.`);
          yield { type: 'info', content: `Nudged: ${openTodos.length} task-list item(s) still open` };
          continue;
        }
        // Daily-driver #1 (self-repair): changed code but never ran it -> force a
        // verify-and-fix turn rather than shipping an unverified edit.
        if (shouldForceVerify({ mode: this.mode, codeChangedUnverified, alreadyForced: forcedVerifyOnce })) {
          forcedVerifyOnce = true;
          this.context.addUser(FORCE_VERIFY_NUDGE);
          yield { type: 'info', content: 'Nudged: verify your code change before finishing' };
          continue;
        }
        // Incomplete extension: a new member added to one enumerated set but not
        // to the others that list the same family. Passing tests do not rule
        // this out — nothing exercises the half that was missed.
        if (this.mode === 'act' && !forcedCompletenessOnce && editedPaths.size > 0) {
          const nudge = await this.buildCompletenessNudgeForTurn(editedPaths);
          // Burn the once-flag only when a nudge is actually ISSUED. Setting it
          // before the check spent the single allowance on the first no-tool-call
          // turn — which is routinely the turn before the gap even exists.
          if (nudge) {
            forcedCompletenessOnce = true;
            this.context.addUser(nudge);
            yield { type: 'info', content: 'Nudged: sibling files may need the same change' };
            continue;
          }
        }
        // Save knowledge state to disk (non-blocking)
        this.context.getKnowledgeState().save().catch((err) => {
          // Losing the knowledge state silently means the next session starts
          // with less than it should and nobody knows why. It must not fail the
          // turn — but it must not be invisible either.
          process.stderr.write(`[knowledge-state] save failed: ${err instanceof Error ? err.message : String(err)}\n`);
        });

        // Keep the latest plan so it survives compaction (memory only).
        this.rememberPlan(fullContent);

        yield* this._fireHooks('Stop', { cwd: process.cwd(), messageCount: this.context.messageCount() });

        this.abortController = null;

        // Restore original model if we switched for vision
        if (visionModelSwitch) {
          this.modelManager.switchTo(visionModelSwitch);
          this.context.setSystemPrompt(visionModelSwitch);
        }

        const tps = evalDuration > 0 ? Math.round((evalCount / evalDuration) * 1e9) : 0;
        yield {
          type: 'done',
          evalCount,
          promptEvalCount,
          tokensPerSecond: tps,
          promptCachedCount,
          promptCachedShare: promptEvalCount > 0 ? promptCachedCount / promptEvalCount : undefined,
        };
        return;
      }

      // Reaching here means the model called a tool -> it took an action this message,
      // so the force-act guard should never fire on a later no-tool-call summary turn.
      hasActedThisMessage = true;
      // Self-repair tracking: a code edit leaves work unverified; a bash run (tests/build/
      // execute) clears it. Walk this turn's calls in order so edit→bash = verified,
      // bash→edit = still unverified.
      // NOTE: the definitive update happens AFTER execution (see
      // `updateVerifyTracking`). This pre-pass is intentionally gone: it walked
      // the REQUESTED calls, so a DENIED edit set "unverified" and earned a
      // spurious nudge, while a denied or FAILING bash cleared it — suppressing
      // the nudge and completing the turn on unverified code, which is the
      // dangerous direction of the two.

      // Snapshot the working tree before anything runs this turn. Placed here
      // rather than at turn start so read-only turns cost nothing.
      await this.checkpointOnce(userMessage);

      // Execute tool calls — parallelize independent read-only calls
      const READ_ONLY_TOOLS = new Set(['read_file', 'glob', 'grep', 'list_files', 'system_info', 'web_search', 'web_fetch', 'bash_output', 'repo_map']);

      // Check if all calls are independent read-only (safe to parallelize)
      // Note: hook plumbing for PreToolUse/PostToolUse is below in both the
      // parallel and sequential paths. See _fireHooks helper at end of class.
      const allReadOnly = toolCalls.length > 1 && toolCalls.every(c => READ_ONLY_TOOLS.has(c.function.name));

      // Per-call result strings for loop signature, in toolCalls order.
      const stepResults: string[] = new Array(toolCalls.length).fill('');
      const stepSuccess: boolean[] = new Array(toolCalls.length).fill(false);

      if (allReadOnly) {
        // Parallel execution for independent read-only calls
        for (const call of toolCalls) {
          yield { type: 'tool_call', name: call.function.name, args: (call.function.arguments || {}) as Record<string, unknown> };
        }
        // Permission checks must be serialized to avoid concurrent prompt races.
        const executableCalls: Array<{ idx: number; name: string; args: Record<string, unknown> }> = [];
        const earlyResults: Array<{ idx: number; name: string; args: Record<string, unknown>; result: ToolResult }> = [];
        for (let i = 0; i < toolCalls.length; i++) {
          const call = toolCalls[i];
          const name = call.function.name;
          const args = (call.function.arguments || {}) as Record<string, unknown>;
          if (allowedTools && !allowedTools.has(name)) {
            earlyResults.push({ idx: i, name, args, result: { success: false, output: '', error: `Tool "${name}" not allowed` } });
            continue;
          }
          const verdict = await this.permissions.checkWithPosture(
            this.postureFor(permissionMode), name, args, undefined, this.verify,
          );
          const decision = typeof verdict === 'string' ? verdict : verdict.decision;
          if (decision === 'deny') {
            // Carry the REASON when there is one. A bare "Permission denied"
            // tells the model nothing it can act on, which is how plan mode
            // used to end with the model quietly reimplementing a tool by hand.
            const error = typeof verdict === 'string' ? 'Permission denied' : verdict.reason;
            earlyResults.push({ idx: i, name, args, result: { success: false, output: '', error } });
            continue;
          }
          // PreToolUse hook — non-zero exit blocks the tool call.
          const preBlock = yield* this._fireHooks('PreToolUse', { tool: name, args, cwd: process.cwd() });
          if (preBlock.blocked) {
            earlyResults.push({ idx: i, name, args, result: { success: false, output: '', error: preBlock.reason || 'Blocked by hook' } });
            continue;
          }
          executableCalls.push({ idx: i, name, args });
        }
        const executed = await Promise.all(executableCalls.map(async ({ idx, name, args }) => {
          const startedAt = Date.now();
          const result = await this.runTool(name, args);
          return { idx, name, args, result, durationMs: Date.now() - startedAt };
        }));
        // Fire PostToolUse for each executed call, in order. Output is purely
        // informational here; PostToolUse cannot abort what already happened.
        for (const { name, args, result, durationMs } of executed) {
          yield* this._fireHooks('PostToolUse', { tool: name, args, cwd: process.cwd(), result, durationMs });
        }
        const results = [...earlyResults, ...executed];
        for (const { idx, name, args, result } of results) {
          yield {
            type: 'tool_result', name, args,
            success: result.success,
            content: result.success ? result.output : result.error,
            error: result.error,
            sources: result.sources,
          };
          const resultContent = result.success ? result.output : `Error: ${result.error}`;
          stepResults[idx] = resultContent;
          this.context.addToolResult(name, resultContent, (args.path as string) || undefined, result.success);
        }
      } else {
        // Sequential execution for write/mixed calls
        for (let i = 0; i < toolCalls.length; i++) {
          const call = toolCalls[i];
          const toolName = call.function.name;
          const toolArgs = (call.function.arguments || {}) as Record<string, unknown>;

          if (allowedTools && !allowedTools.has(toolName) && (toolName !== 'update_memory' || this.registry.policy)) {
            const msg = `Tool "${toolName}" not allowed`;
            yield { type: 'tool_result', name: toolName, success: false, content: `Tool "${toolName}" is not in the allowed set for this request` };
            stepResults[i] = msg;
            this.context.addToolResult(toolName, msg, undefined, false);
            continue;
          }

          if (toolName === 'update_memory') {
            const key = (toolArgs.key as string) || '';
            const value = (toolArgs.value as string) || '';
            this.context.getKnowledgeState().updateMemory(key, value);
            const msg = `Stored: ${key} = ${value}`;
            yield { type: 'tool_result', name: toolName, success: true, content: msg };
            stepResults[i] = msg;
            this.context.addToolResult(toolName, msg);
            continue;
          }

          yield { type: 'tool_call', name: toolName, args: toolArgs };

          const preview = this._previewToolCall(toolName, toolArgs);
          const verdict = await this.permissions.checkWithPosture(
            this.postureFor(permissionMode), toolName, toolArgs, preview, this.verify,
          );
          const decision = typeof verdict === 'string' ? verdict : verdict.decision;
          if (decision === 'deny') {
            yield { type: 'permission_denied', name: toolName };
            const msg = typeof verdict === 'string'
              ? `Permission denied: user rejected ${toolName}`
              : verdict.reason;
            stepResults[i] = msg;
            this.context.addToolResult(toolName, msg, undefined, false);
            continue;
          }

          // PreToolUse hook — non-zero exit blocks the tool call.
          const preBlock = yield* this._fireHooks('PreToolUse', {
            tool: toolName, args: toolArgs, cwd: process.cwd(),
          });
          if (preBlock.blocked) {
            const msg = preBlock.reason || 'Blocked by hook';
            yield {
              type: 'tool_result',
              name: toolName,
              success: false,
              content: msg,
              error: msg,
            };
            stepResults[i] = msg;
            this.context.addToolResult(toolName, msg, (toolArgs.path as string) || undefined, false);
            continue;
          }

          const startedAt = Date.now();
          const result = await this.runTool(toolName, toolArgs);
          const durationMs = Date.now() - startedAt;

          // PostToolUse hook — informational; cannot abort.
          yield* this._fireHooks('PostToolUse', {
            tool: toolName, args: toolArgs, cwd: process.cwd(), result, durationMs,
          });

          yield {
            type: 'tool_result',
            name: toolName,
            args: toolArgs,
            success: result.success,
            content: result.success ? result.output : result.error,
            error: result.error,
            sources: result.sources,
          };

          // Only SUCCESSFUL writes count as "the model handled this file". An
          // attempted-but-failed edit must stay eligible for the completeness
          // nudge — a file the model tried and could not change is exactly the
          // one most likely to have been left behind.
          if (result.success && CODE_MUTATION_TOOLS.has(toolName)) {
            const ep = (toolArgs as { path?: string }).path;
            if (typeof ep === 'string' && ep) editedPaths.add(resolve(ep));
          }

          // Self-repair tracking, from what ACTUALLY happened. A successful code
          // change leaves work unverified; a bash run that could plausibly check
          // it clears that. Both halves require success — a failing test run does
          // not verify anything, and it is the one most likely to be mistaken for
          // verification.
          if (result.success && CODE_MUTATION_TOOLS.has(toolName)) {
            codeChangedUnverified = true;
          } else if (result.success && toolName === 'bash'
                     && bashVerifies(String((toolArgs as { command?: unknown }).command ?? ''))) {
            codeChangedUnverified = false;
          }

          const resultContent = result.success ? result.output : `Error: ${result.error}`;
          const filePath = (toolArgs.path as string) || undefined;
          stepResults[i] = resultContent;
          stepSuccess[i] = result.success;
          this.context.addToolResult(toolName, resultContent, filePath, result.success);
        }
      }

      // Flush knowledge state update after all tool results are collected
      this.context.flushKnowledgeUpdate(fullContent);

      // Stuck loop detection: signature = sha256(name + args + result) per call.
      // Same call + same output > LOOP_MAX_REPEATS times in a LOOP_WINDOW window
      // means stuck. Same call + different output is productive iteration.
      if (toolCalls.length > 0) {
        const sig = signatureOf(toolCalls, stepResults);
        if (sig) {
          recentSteps.push({
            signature: sig,
            callSignature: callSignatureOf(toolCalls),
            allFailed: stepSuccess.every(ok => !ok),
            mutated: toolCalls.some((c, idx) =>
              stepSuccess[idx] && CODE_MUTATION_TOOLS.has(c.function.name)),
          });
          if (recentSteps.length > LOOP_WINDOW) recentSteps.shift();
          const names = toolCalls.map(c => c.function.name).join(', ');
          if (detectStuckSignature(recentSteps)) {
            yield { type: 'error', error: `Stopped: same tool call+result repeated >${LOOP_MAX_REPEATS} times in last ${LOOP_WINDOW} steps (${names}). Likely stuck.` };
            this.abortController = null;
            return;
          }
          // The loop the byte-identical check cannot see: the same call failing
          // over and over with slightly different error text each time.
          //
          // WARN BEFORE KILLING. Whether this is a loop or a debugging cycle is
          // an INFERENCE from an ambiguous signal, and this guard has already
          // been wrong once tonight — it stopped an agent 34 tool calls into a
          // real multi-file change for running `npm test` three times. A guard
          // may not take a terminal action on an inferred state; it gets to say
          // so, and the model gets to answer. Only a second occurrence AFTER the
          // warning is evidence rather than a guess.
          if (detectRepeatedFailure(recentSteps)) {
            if (!repeatedFailureWarned) {
              repeatedFailureWarned = true;
              recentSteps.length = 0;
              this.context.addUser(
                `[SYSTEM] ${names} has now failed ${REPEATED_FAILURE_LIMIT} times with identical arguments and nothing changed in between. `
                + `If you are iterating on a fix, keep going — this is only a warning. If you are repeating a call that cannot work, `
                + `change the approach: read the file again, check the path, or try a different tool.`,
              );
              yield { type: 'info', content: 'Warned: repeated identical failure' };
            } else {
              yield { type: 'error', error: `Stopped: ${names} kept failing with identical arguments after a warning. Likely stuck.` };
              this.abortController = null;
              return;
            }
          }
        }
      }

      // Detect turns with no user-visible content
      if (!fullContent.replace(/<think>[\s\S]*?<\/think>/g, '').trim()) {
        turnsWithoutUserContent++;
        if (turnsWithoutUserContent >= MAX_TURNS_WITHOUT_OUTPUT) {
          yield { type: 'error', error: `Stopped: ${MAX_TURNS_WITHOUT_OUTPUT} turns with no visible output. The model may be stuck.` };
          this.abortController = null;
          return;
        }
      } else {
        turnsWithoutUserContent = 0;
      }

      // Proactive compaction check after tool results (context grows most here)
      if (this.context.needsCompaction()) {
        const retryEvents: Array<{ attempt: number; projected: number; limit: number }> = [];
        yield* this._fireHooks('PreCompact', { cwd: process.cwd(), messageCount: this.context.messageCount() });
        const compacted = await this.context.compactWithRetry(
          createChatClient(this.config),
          this.modelManager.getCurrentModel(),
          this.config.summarizerModel,
          {
            onRetry: (attempt, projected, limit) => {
              retryEvents.push({ attempt, projected, limit });
            },
          },
        );
        if (compacted) {
          yield { type: 'info', content: 'Compacted conversation to free context space' };
          for (const r of retryEvents) {
            yield { type: 'info', content: `Compacting harder (attempt ${r.attempt}) — projected ${r.projected} > ${Math.round(r.limit * 0.85)} cutoff` };
          }

          if (this.sessionPlan) {
            this.context.addUser(RESTORED_PLAN_NOTE + this.sessionPlan);
            yield { type: 'info', content: "Restored this session's plan after compaction" };
          }
        }
      }
      }
    } finally {
      this._runResult = {
        editedPaths: new Set(editedPaths),
        codeChangedUnverified,
      };
    }

    // NOTE: no Stop hook here. The turn loop above is `for (;;)` with no
    // `break` — every exit is a `return`, so this point is unreachable. The
    // Stop hook fires in the `toolCalls.length === 0` branch, which is how
    // *every* successful run terminates (tool-using turns simply loop until
    // the model stops calling tools and then land there too).
  }

  /** Compute a preview string for a tool call that mutates files. Returns
   *  undefined for tools we don't preview, or when the preview can't be
   *  computed (e.g. file doesn't exist for an edit). The preview is shown
   *  in the permission prompt so the user can approve with full context.
   */
  private _previewToolCall(toolName: string, args: Record<string, unknown>): string | undefined {
    try {
      const path = typeof args.path === 'string' ? resolve(args.path) : null;
      if (!path) return undefined;

      if (toolName === 'edit_file') {
        if (!existsSync(path)) return undefined;
        const oldContent = readFileSync(path, 'utf-8');
        const oldStr = String(args.old_string ?? '');
        const newStr = String(args.new_string ?? '');
        const replaceAll = args.replace_all === true;
        if (!oldContent.includes(oldStr)) return undefined;
        const newContent = replaceAll
          ? oldContent.split(oldStr).join(newStr)
          : oldContent.replace(oldStr, newStr);
        return previewEdit(oldContent, newContent, relative(process.cwd(), path));
      }

      if (toolName === 'write_file') {
        const newContent = typeof args.content === 'string' ? args.content : '';
        const existing = existsSync(path) ? readFileSync(path, 'utf-8') : null;
        return previewWrite(existing, newContent, relative(process.cwd(), path));
      }

      if (toolName === 'multi_edit') {
        if (!existsSync(path)) return undefined;
        const oldContent = readFileSync(path, 'utf-8');
        const edits = Array.isArray(args.edits) ? args.edits as Array<{ old_string?: string; new_string?: string; replace_all?: boolean }> : [];
        // Best-effort simulation: apply edits sequentially with simple
        // replace; if any step doesn't match, bail and skip preview.
        let working = oldContent;
        for (const e of edits) {
          const oldStr = String(e.old_string ?? '');
          const newStr = String(e.new_string ?? '');
          if (!working.includes(oldStr)) return undefined;
          working = e.replace_all === true
            ? working.split(oldStr).join(newStr)
            : working.replace(oldStr, newStr);
        }
        return previewEdit(oldContent, working, relative(process.cwd(), path));
      }

      return undefined;
    } catch {
      return undefined; // never block on preview failure
    }
  }

  /** Fire all matching hooks for a lifecycle event. Yields a `hook_output`
   *  event for each hook whose stdout, stderr, or non-zero exit is worth
   *  surfacing to the user. Returns the block decision so callers can abort
   *  the action (PreToolUse semantics). For events that never block
   *  (PostToolUse, Stop, Notification, UserPromptSubmit), the return value
   *  is harmlessly ignored.
   *
   *  Use `yield* this._fireHooks(event, payload)` from the calling generator.
   */
  private async *_fireHooks(
    event: HookEventName,
    payload: Record<string, unknown>,
  ): AsyncGenerator<AgentEvent, { blocked: boolean; reason?: string }> {
    const results: HookExecResult[] = this.registry.policy ? [] : await runHooks(event, payload as never);
    for (const r of results) {
      const text = r.stdout || r.stderr;
      const blockedHere = event === 'PreToolUse' && r.exitCode !== 0;
      // Surface output only when there's something to show, the hook timed
      // out, or the hook is going to block — don't pollute the chat with
      // silent successful hooks.
      if (text || r.timedOut || blockedHere) {
        const content = r.timedOut
          ? `[hook ${event}] timed out: ${r.hook.command}`
          : (text || `[hook ${event}] exited ${r.exitCode}`);
        yield {
          type: 'hook_output',
          content,
          hookEvent: event,
          hookLayer: r.layer,
          hookExitCode: r.exitCode,
          hookBlocked: blockedHere,
        };
      }
    }
    return shouldBlock(results);
  }

  /** Non-streaming version for API use (no permission prompts — auto-allows) */
  async runSync(
    userMessage: string,
    options?: { permissionMode?: PermissionMode; allowedTools?: string[] | null },
  ): Promise<{
    content: string;
    toolCalls: Array<{ name: string; args: Record<string, unknown>; result: string; success: boolean }>;
    errors: string[];
    stuck: boolean;
  }> {
    let content = '';
    const toolCallResults: Array<{ name: string; args: Record<string, unknown>; result: string; success: boolean }> = [];
    const errors: string[] = [];
    let stuck = false;

    for await (const event of this.run(userMessage, options)) {
      switch (event.type) {
        case 'text':
          content += event.content || '';
          break;
        case 'tool_result':
          toolCallResults.push({
            name: event.name || '',
            args: event.args || {},
            result: event.content || event.error || '',
            success: event.success !== false,
          });
          break;
        case 'error': {
          const em = String(event.error || event.content || '');
          errors.push(em);
          if (/stuck|no progress|no output|repeat/i.test(em)) stuck = true;
          break;
        }
      }
    }

    return { content, toolCalls: toolCallResults, errors, stuck };
  }

  clear(): void {
    this.context.clear();
    this.context.setSystemPrompt(this.modelManager.getCurrentModel());
  }
}
