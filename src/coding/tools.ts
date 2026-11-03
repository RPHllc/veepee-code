/** Audited coding profile tools; never enroll upstream bash/git/github implementations. */
import { constants, lstatSync, openSync, closeSync, readFileSync, writeFileSync, readdirSync, mkdirSync, renameSync, unlinkSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { z } from 'zod';
import type { ToolDef, ToolResult } from '../tools/types.js';

export const CODING_TOOLS = ['read_file', 'list_files', 'search_files', 'write_file', 'edit_file', 'run_command', 'git_status', 'git_diff'];
export type RPC = (method: 'run_command' | 'git_status' | 'git_diff', params: Record<string, unknown>) => Promise<ToolResult>;
const pathSchema = z.string().min(1).max(240);
const secretName = /^(?:\.env(?:\..*)?|id_rsa|id_ed25519|credentials|secrets(?:\..*)?)$/i;

export function safePath(root: string, path: string, allowMissing = false): string {
  if (!/^[a-zA-Z0-9_./ -]+$/.test(path) || path.startsWith('/') || path.endsWith('/')) throw new Error('Workspace-relative path required');
  let current = root;
  const parts = path.split('/');
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part || part.startsWith('-') || part === '.' || part === '..' || part.toLowerCase() === '.git' || secretName.test(part)) throw new Error('Protected or escaping path');
    current = join(current, part);
    let stat;
    try { stat = lstatSync(current); } catch (error) {
      if (allowMissing && (error as NodeJS.ErrnoException).code === 'ENOENT') continue;
      throw error;
    }
    if (stat.isSymbolicLink() || (!stat.isDirectory() && (!stat.isFile() || stat.nlink !== 1))) throw new Error('Links and special files are forbidden');
    if (i < parts.length - 1 && !stat.isDirectory()) throw new Error('Not a workspace directory');
  }
  return current;
}

export function codingTools(root: string, commands: string[], rpc: RPC): ToolDef[] {
  // The upstream agent can run tool batches concurrently. Serialize every workspace
  // operation, including the complete executor lifetime, so hostile test code cannot
  // race an agent read/write after path validation.
  let tail: Promise<unknown> = Promise.resolve();
  const tool = (name: string, description: string, schema: z.ZodObject<any>, execute: ToolDef['execute']): ToolDef => ({
    name, description, schema, timeoutMs: null, verifiesWorkspace: name === 'run_command',
    execute(params) {
      const pending = tail.then(async () => {
        try { return await execute(params); }
        catch { return { success: false, output: '', error: 'Coding workspace operation rejected' }; }
      });
      tail = pending.catch(() => undefined); return pending;
    },
  });
  const read = (path: string) => {
    const p = safePath(root, path); const stat = lstatSync(p);
    if (!stat.isFile() || stat.size > 262144) throw new Error('Bounded regular file required');
    const fd = openSync(p, constants.O_RDONLY | constants.O_NOFOLLOW);
    try { return readFileSync(fd, 'utf8'); } finally { closeSync(fd); }
  };
  const write = (path: string, content: string) => {
    const p = safePath(root, path, true);
    if (Buffer.byteLength(content) > 262144) throw new Error('File too large');
    mkdirSync(dirname(p), { recursive: true });
    safePath(root, path, true);
    const tmp = p + '.coding-next'; safePath(root, path + '.coding-next', true);
    const fd = openSync(tmp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o644);
    try { writeFileSync(fd, content); } finally { closeSync(fd); }
    try { renameSync(tmp, p); } catch (e) { unlinkSync(tmp); throw e; }
  };
  const files = () => {
    const result: string[] = [];
    const visit = (path = '') => {
      for (const entry of readdirSync(path ? safePath(root, path) : root, { withFileTypes: true })) {
        const name = path ? path + '/' + entry.name : entry.name;
        safePath(root, name);
        if (entry.isDirectory()) visit(name);
        else result.push(name);
        if (result.length > 512) throw new Error('File inventory limit exceeded');
      }
    };
    visit(); return result.sort();
  };
  return [
    tool('read_file', 'Read one bounded UTF-8 workspace file; links and secrets are denied.', z.object({ path: pathSchema }).strict(), async p => ({ success: true, output: read(String(p.path)) })),
    tool('list_files', 'List relative workspace files. No host or Git metadata access.', z.object({}).strict(), async () => ({ success: true, output: files().join('\n') })),
    tool('search_files', 'Literal text search, bounded to 100 matching lines. Not a regex or shell.', z.object({ text: z.string().min(1).max(100) }).strict(), async p => {
      const result: string[] = [];
      for (const file of files()) {
        for (const [i, line] of read(file).split('\n').entries()) {
          if (line.includes(String(p.text))) result.push(`${file}:${i + 1}:${line.slice(0, 300)}`);
          if (result.length === 100) return { success: true, output: result.join('\n') };
        }
      }
      return { success: true, output: result.join('\n') };
    }),
    tool('write_file', 'Create or replace a workspace file. No links, host paths or Git metadata.', z.object({ path: pathSchema, content: z.string().max(262144) }).strict(), async p => { write(String(p.path), String(p.content)); return { success: true, output: 'Workspace file written' }; }),
    tool('edit_file', 'Replace one exact, unique literal occurrence in a workspace file.', z.object({ path: pathSchema, old_string: z.string().min(1).max(262144), new_string: z.string().max(262144) }).strict(), async p => {
      const old = read(String(p.path)), match = String(p.old_string);
      if (old.split(match).length !== 2) throw new Error('Unique exact match required');
      write(String(p.path), old.replace(match, String(p.new_string))); return { success: true, output: 'Workspace file edited' };
    }),
    tool('run_command', 'Run a server-approved test profile in a separate networkless container. Available IDs: ' + commands.join(', '), z.object({ command_id: z.string().min(1).max(64) }).strict(), async p => {
      if (!commands.includes(String(p.command_id))) return { success: false, output: '', error: 'Command profile denied' };
      return rpc('run_command', p);
    }),
    tool('git_status', 'Inspect actual local Git status against the exact base; no Git arguments or mutations.', z.object({}).strict(), async () => rpc('git_status', {})),
    tool('git_diff', 'Inspect actual local Git diff against the exact base; no Git arguments or mutations.', z.object({}).strict(), async () => rpc('git_diff', {})),
  ];
}
