import { readFileSync } from 'node:fs';
import { z } from 'zod';

const schema = z.object({
  schema_version: z.literal(1),
  role: z.string().min(1).max(64),
  deny_unlisted_tools: z.literal(true),
  forbid_direct_tool_execute: z.literal(true),
  allowed_tools: z.array(z.string().regex(/^[a-z][a-z0-9_]*$/)).max(64),
  max_tool_calls_per_job: z.number().int().min(1).max(64),
  max_job_seconds: z.number().int().min(1).max(180),
}).strict();

/** Immutable, operator-owned ceiling. A client may only narrow this set. */
export class WorkerPolicy {
  private readonly names: ReadonlySet<string>;
  readonly maxToolCalls: number;
  readonly maxTurns: number;
  readonly deadlineMs: number;
  constructor(value: unknown) {
    const parsed = schema.parse(value);
    if (new Set(parsed.allowed_tools).size !== parsed.allowed_tools.length) throw new Error('Duplicate policy tool');
    this.names = new Set(parsed.allowed_tools);
    this.maxToolCalls = parsed.max_tool_calls_per_job;
    this.maxTurns = Math.min(this.maxToolCalls + 2, 20);
    this.deadlineMs = parsed.max_job_seconds * 1000;
    Object.freeze(this);
  }
  allows(name: string): boolean { return this.names.has(name); }
  intersect(client?: string[] | null): string[] {
    return [...this.names].filter(n => client == null || client.includes(n));
  }
  validateAvailable(names: string[]): void {
    for (const n of this.names) if (!names.includes(n)) throw new Error(`Unenforceable policy tool: ${n}`);
  }
  static load(path: string): WorkerPolicy {
    if (!path) throw new Error('Worker policy path is required');
    return new WorkerPolicy(JSON.parse(readFileSync(path, 'utf8')));
  }
}
