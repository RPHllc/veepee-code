import { describe, it, expect } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, symlinkSync, linkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { codingTools, CODING_TOOLS } from '../src/coding/tools.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { WorkerPolicy } from '../src/worker-policy.js';
import { ContextManager } from '../src/context.js';
const data = { schema_version: 1, role: 'coding', deny_unlisted_tools: true, forbid_direct_tool_execute: true, allowed_tools: CODING_TOOLS, max_tool_calls_per_job: 32, max_job_seconds: 60 };
function harness(rpc = async () => ({ success: true, output: 'real process record' })) {
  const root = mkdtempSync(join(tmpdir(), 'coding-unit-'));
  const registry = new ToolRegistry(new WorkerPolicy(data));
  for (const t of codingTools(root, ['python_unittest'], rpc)) registry.register(t);
  return { root, registry, close: () => rmSync(root, { recursive: true, force: true }) };
}
describe('isolated coding tools reuse policy execution boundary', () => {
  it.each(['bash', 'git', 'github', 'docker', 'http_request', 'web_search', 'task', 'mcp_write', 'remote_execute', 'todo_write', 'update_memory'])('rejects unapproved %s', async name => {
    const h = harness(); try { expect((await h.registry.execute(name, {})).success).toBe(false); } finally { h.close(); }
  });
  it.each(['/etc/passwd', '../outside', 'a/../../outside', '.git/config', '.env', 'keys/id_rsa', 'a//file', 'x/./file'])('rejects %s', async path => {
    const h = harness(); try { expect((await h.registry.execute('write_file', { path, content: 'bad' })).success).toBe(false); } finally { h.close(); }
  });
  it('edits regular files, searches literally, and refuses links/ambiguous edits', async () => {
    const h = harness();
    try {
      expect((await h.registry.execute('write_file', { path: 'src/main.py', content: 'return a - b' })).success).toBe(true);
      expect((await h.registry.execute('edit_file', { path: 'src/main.py', old_string: 'a - b', new_string: 'a + b' })).success).toBe(true);
      expect(readFileSync(join(h.root, 'src/main.py'), 'utf8')).toContain('a + b');
      expect((await h.registry.execute('search_files', { text: 'a + b' })).output).toContain('src/main.py:1');
      symlinkSync('/etc', join(h.root, 'escape'));
      expect((await h.registry.execute('read_file', { path: 'escape/passwd' })).success).toBe(false);
      linkSync(join(h.root, 'src/main.py'), join(h.root, 'hard'));
      expect((await h.registry.execute('read_file', { path: 'hard' })).success).toBe(false);
      expect((await h.registry.execute('edit_file', { path: 'src/main.py', old_string: 'not-there', new_string: 'bad' })).success).toBe(false);
    } finally { h.close(); }
  });
  it('serializes edits behind the whole executor lifetime and denies arbitrary commands', async () => {
    let release!: () => void; let entered!: () => void;
    const started = new Promise<void>(r => { entered = r; });
    const gate = new Promise<void>(r => { release = r; });
    const h = harness(async () => { entered(); await gate; return { success: true, output: 'exit=0' }; });
    try {
      const command = h.registry.execute('run_command', { command_id: 'python_unittest' }); await started;
      let done = false; const write = h.registry.execute('write_file', { path: 'file', content: 'yes' }).then(v => { done = true; return v; });
      await new Promise(r => setTimeout(r, 5)); expect(done).toBe(false);
      release(); await command; expect((await write).success).toBe(true);
      expect((await h.registry.execute('run_command', { command_id: 'bash' })).success).toBe(false);
      expect(h.registry.get('run_command')!.verifiesWorkspace).toBe(true);
    } finally { h.close(); }
  });
  it('supports a trusted isolated prompt without changing the research default', () => {
    const research = new ContextManager(undefined, true); research.setSystemPrompt('test');
    const coding = new ContextManager(undefined, true, 'Trusted coding profile'); coding.setSystemPrompt('test');
    expect(research.getSystemPrompt()).toContain('bounded research worker');
    expect(coding.getSystemPrompt()).toBe('Trusted coding profile');
  });
});
