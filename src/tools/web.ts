import { z } from 'zod';
import type { ToolDef } from './types.js';
import { ok, fail } from './types.js';
import type { Config } from '../config.js';

/** Hard ceiling on a fetched body, applied while streaming rather than after.
 *  `res.text()` / `res.json()` buffer the WHOLE response first, so a large log
 *  or a binary artifact could exhaust memory before any truncation ran. */
const MAX_RESPONSE_BYTES = 5 * 1024 * 1024;

/** Read a response body, stopping once the cap is reached. */
async function readCapped(res: Response, limit = MAX_RESPONSE_BYTES): Promise<string> {
  const body = res.body;
  if (!body) return '';
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const chunks: string[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > limit) {
        chunks.push(decoder.decode(value.slice(0, Math.max(0, value.byteLength - (total - limit))), { stream: false }));
        chunks.push(`\n…(response truncated at ${Math.round(limit / 1024 / 1024)}MB)`);
        break;
      }
      chunks.push(decoder.decode(value, { stream: true }));
    }
  } finally {
    try { await reader.cancel(); } catch { /* already closed */ }
  }
  return chunks.join('');
}

export function registerWebTools(config: Config): ToolDef[] {
  const tools: ToolDef[] = [
    createWebFetchTool(config.agentlensUrl),
    createHttpRequestTool(),
  ];

  if (config.searxngUrl) {
    tools.push(createWebSearchTool(config.searxngUrl));
  }

  return tools;
}

/** Extract a page via agentlens (`GET /parse?url=`), returning token-efficient
 *  readable content. Returns null on any failure so the caller can fall back to
 *  a raw fetch. agentlens response shape: { title, content, token_estimate, ... }. */
async function agentlensParse(agentlensUrl: string, url: string): Promise<string | null> {
  try {
    const base = agentlensUrl.replace(/\/+$/, '');
    const res = await fetch(`${base}/parse?url=${encodeURIComponent(url)}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) return null;
    const data = await res.json() as { title?: string | null; content?: string | null };
    const content = (data.content || '').trim();
    if (!content) return null;
    const title = (data.title || '').trim();
    return title ? `# ${title}\n\n${content}` : content;
  } catch {
    return null;
  }
}

/** Raw fetch + local HTML strip. Fallback when agentlens is unset/unreachable. */
async function rawFetchText(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: {
      'User-Agent': 'veepee-code/0.3 (CLI coding assistant)',
      'Accept': 'text/html,application/xhtml+xml,application/json,text/plain',
    },
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return JSON.stringify(JSON.parse(await readCapped(res)), null, 2);
  }

  const html = await readCapped(res);
  // An empty body is a FACT about the fetch, not a page with no words in it.
  // Returned as '', a 200 with no body, a JS-only shell and a stripped-to-
  // nothing page are indistinguishable from each other and from a successful
  // read — so the model concludes the page said nothing and moves on.
  if (!html.trim()) {
    return `(the response had no readable body — ${res.status} ${res.statusText || ''}, content-type "${contentType || 'unset'}". `
      + `This is not the same as a page with no content: it may be JavaScript-rendered, or served empty to a non-browser client.)`;
  }
  // Basic HTML stripping — remove tags, decode entities, collapse whitespace
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
    .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
    .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function createWebFetchTool(agentlensUrl: string | null): ToolDef {
  return {
    name: 'web_fetch',
    description: 'Fetch a web page and extract its readable text content. Uses agentlens for token-efficient extraction when configured (falls back to a raw fetch). Good for reading documentation, articles, or API responses.',
    schema: z.object({
      url: z.string().describe('The URL to fetch'),
      max_length: z.number().optional().describe('Maximum characters to return (default 10000)'),
    }),
    execute: async (params) => {
      try {
        const url = params.url as string;
        const maxLen = (params.max_length as number) || 10000;

        // Prefer agentlens (token-efficient extraction). Fall back to a raw
        // fetch + local HTML strip if it's unset or unreachable.
        let text: string | null = agentlensUrl ? await agentlensParse(agentlensUrl, url) : null;
        if (text === null) {
          text = await rawFetchText(url);
        }

        if (text.length > maxLen) {
          text = text.slice(0, maxLen) + `\n... (truncated at ${maxLen} chars)`;
        }

        return ok(text);
      } catch (err) {
        return fail(`Fetch failed: ${(err as Error).message}`);
      }
    },
  };
}

function createHttpRequestTool(): ToolDef {
  return {
    name: 'http_request',
    description: 'Make an HTTP request with full control over method, headers, and body. Use for API calls.',
    schema: z.object({
      url: z.string().describe('The URL to request'),
      method: z.enum(['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD']).optional().describe('HTTP method (default GET)'),
      headers: z.string().optional().describe('Headers as JSON string, e.g. {"Authorization": "Bearer xxx"}'),
      body: z.string().optional().describe('Request body (string or JSON string)'),
    }),
    execute: async (params) => {
      try {
        const method = (params.method as string) || 'GET';
        let headers: Record<string, string> = {};

        if (params.headers) {
          try {
            headers = JSON.parse(params.headers as string);
          } catch {
            return fail('Invalid headers JSON');
          }
        }

        const fetchOpts: RequestInit = {
          method,
          headers,
          signal: AbortSignal.timeout(30000),
        };

        if (params.body && method !== 'GET' && method !== 'HEAD') {
          fetchOpts.body = params.body as string;
          if (!headers['Content-Type'] && !headers['content-type']) {
            (fetchOpts.headers as Record<string, string>)['Content-Type'] = 'application/json';
          }
        }

        const res = await fetch(params.url as string, fetchOpts);
        const contentType = res.headers.get('content-type') || '';
        let body: string;

        if (contentType.includes('json')) {
          const raw = await readCapped(res);
          try { body = JSON.stringify(JSON.parse(raw), null, 2); } catch { body = raw; }
        } else {
          body = await readCapped(res);
        }

        const statusLine = `HTTP ${res.status} ${res.statusText}`;
        if (body.length > 10000) {
          body = body.slice(0, 10000) + '\n... (truncated)';
        }

        return ok(`${statusLine}\n\n${body}`);
      } catch (err) {
        return fail(`Request failed: ${(err as Error).message}`);
      }
    },
  };
}

export function createWebSearchTool(searxngUrl: string): ToolDef {
  return {
    name: 'web_search',
    description: 'Search the web using SearXNG. Returns search results with titles, URLs, and snippets. Good for finding documentation, Stack Overflow answers, or current information.',
    schema: z.object({
      query: z.string().trim().min(1).max(512).describe('Search query'),
      max_results: z.number().int().min(1).max(10).optional().describe('Maximum results to return (default 5)'),
    }),
    execute: async (params) => {
      try {
        const maxResults = (params.max_results as number) || 5;
        const base = searxngUrl.replace(/\/+$/, '');
        const url = `${base}/search?q=${encodeURIComponent(params.query as string)}&format=json&language=en`;

        const res = await fetch(url, { signal: AbortSignal.timeout(15000), redirect: 'error', headers: { 'User-Agent': 'veepee-code/0.3' } });
        if (!res.ok) return fail(`Search API returned ${res.status}`);

        const data = JSON.parse(await readCapped(res, 256 * 1024)) as { results: Array<{ title: string; url: string; content: string }> };
        if (!Array.isArray(data.results)) return fail('Invalid search response');
        const results = data.results.filter(r => {
          try { const u = new URL(r.url); return ['http:', 'https:'].includes(u.protocol) && !u.username && !u.password; } catch { return false; }
        }).slice(0, maxResults);

        if (results.length === 0) return ok('No results found.');

        const output = results
          .map((r, i) => `${i + 1}. ${String(r.title || '').slice(0, 300)}\n   ${r.url.slice(0, 2048)}\n   ${String(r.content || '(no snippet)').slice(0, 2000)}`)
          .join('\n\n');

        return { ...ok(output), sources: results.map(r => r.url).filter(u => Buffer.byteLength(u) <= 512) };
      } catch (err) {
        return fail('Search failed: unavailable or invalid bounded response');
      }
    },
  };
}
