import type { Tool as OllamaTool } from 'ollama';
import { z } from 'zod';

export interface ToolResult {
  success: boolean;
  output: string;
  error?: string;
  sources?: string[];
}

/** Where a tool came from. Drives `/tools` grouping, lint coverage, and
 *  diagnostic surface (e.g. "[mcp:filesystem]" prefix on errors). */
export type ToolSource = 'local' | 'remote' | 'mcp' | 'skill';

export interface ToolDef {
  /** Trusted adapter: a successful result represents actual workspace verification. */
  verifiesWorkspace?: boolean;
  name: string;
  description: string;
  schema: z.ZodObject<z.ZodRawShape>;
  execute: (params: Record<string, unknown>) => Promise<ToolResult>;
  /** Optional provenance — defaults to 'local' when omitted (preserves
   *  existing behavior for the ~15 hand-written local tools). Remote bridge,
   *  MCP client, and skill loader set this explicitly. */
  source?: ToolSource;
  /** For non-local sources, an optional sub-name (e.g. MCP server name,
   *  skill file basename). Used for grouping in `/tools`. */
  sourceName?: string;
  /** Wall-clock budget for one execute() call, enforced by ToolRegistry.
   *  Omit to take the default; set `null` for tools that legitimately run
   *  unbounded (subagents, deep research). */
  timeoutMs?: number | null;
}

/** Convert a ToolDef's Zod schema to Ollama tool format */
export function toOllamaTool(tool: ToolDef): OllamaTool {
  const shape = tool.schema.shape;
  const properties: Record<string, { type: string; description?: string; enum?: string[] }> = {};
  const required: string[] = [];

  for (const [key, zodType] of Object.entries(shape)) {
    const field = zodType as z.ZodTypeAny;
    let innerType = field;
    let isOptional = false;

    // Unwrap optional
    if (innerType instanceof z.ZodOptional) {
      isOptional = true;
      innerType = innerType.unwrap();
    }

    // Unwrap default
    if (innerType instanceof z.ZodDefault) {
      innerType = innerType.removeDefault();
    }

    const prop: { type: string; description?: string; enum?: string[] } = { type: 'string' };

    if (innerType instanceof z.ZodString) {
      prop.type = 'string';
    } else if (innerType instanceof z.ZodNumber) {
      prop.type = 'number';
    } else if (innerType instanceof z.ZodBoolean) {
      prop.type = 'boolean';
    } else if (innerType instanceof z.ZodEnum) {
      prop.type = 'string';
      prop.enum = innerType.options as string[];
    } else if (innerType instanceof z.ZodArray) {
      prop.type = 'array';
    } else {
      prop.type = 'string';
    }

    // Extract description from Zod metadata
    if (field.description) {
      prop.description = field.description;
    }

    properties[key] = prop;
    if (!isOptional) {
      required.push(key);
    }
  }

  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: 'object',
        properties,
        required,
      },
    },
  };
}

export function ok(output: string): ToolResult {
  return { success: true, output };
}

export function fail(error: string): ToolResult {
  return { success: false, output: '', error };
}
