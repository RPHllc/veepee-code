import { describe, it, expect } from 'vitest';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, symlinkSync, linkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { codingTools, CODING_TOOLS } from '../src/coding/tools.js';
import { ToolRegistry } from '../src/tools/registry.js';
import { WorkerPolicy } from '../src/worker-policy.js';
import { ContextManager } from '../src/context.js';
const data = { schema_version: 1, role: 'coding', deny_unlisted_tools: true, forbid_direct_tool_execute: true, allowed_tools: CODING_TOOLS, max_tool_calls_per_job: 32, max_job_seconds: 60 };
function harness(rpc = async () => ({ success: true, output: 'real process record' }), writablePaths?: string[]) {
  const root = mkdtempSync(join(tmpdir(), 'coding-unit-'));
  const registry = new ToolRegistry(new WorkerPolicy(data));
  try {
    for (const t of codingTools(root, ['python_unittest'], rpc, writablePaths)) registry.register(t);
  } catch (error) { rmSync(root, { recursive: true, force: true }); throw error; }
  return { root, registry, close: () => rmSync(root, { recursive: true, force: true }) };
}
describe('isolated coding tools reuse policy execution boundary', () => {
  it('enforces exact enrolled writes while allowing bounded reads of protected repository files', async () => {
    const h = harness(undefined, ['app.py', 'src/new.py']);
    try {
      mkdirSync(join(h.root, 'tests'));
      writeFileSync(join(h.root, 'README.md'), 'protected requirements');
      writeFileSync(join(h.root, 'tests/test_app.py'), 'protected test');
      writeFileSync(join(h.root, 'app.py'), 'return a - b');
      for (const path of ['README.md', 'tests/test_app.py', 'app.py.bak', 'app.py.coding-next', './app.py', 'src/../app.py']) {
        expect((await h.registry.execute('write_file', { path, content: 'unauthorized' })).success).toBe(false);
      }
      expect((await h.registry.execute('edit_file', { path: 'tests/test_app.py', old_string: 'protected test', new_string: 'pass' })).success).toBe(false);
      expect((await h.registry.execute('read_file', { path: 'README.md' })).output).toBe('protected requirements');
      expect((await h.registry.execute('read_file', { path: 'tests/test_app.py' })).output).toBe('protected test');
      expect((await h.registry.execute('edit_file', { path: 'app.py', old_string: 'a - b', new_string: 'a + b' })).success).toBe(true);
      expect((await h.registry.execute('write_file', { path: 'src/new.py', content: 'allowed source' })).success).toBe(true);
      expect(readFileSync(join(h.root, 'app.py'), 'utf8')).toBe('return a + b');
      expect(readFileSync(join(h.root, 'tests/test_app.py'), 'utf8')).toBe('protected test');
      expect(readFileSync(join(h.root, 'README.md'), 'utf8')).toBe('protected requirements');
      expect((await h.registry.execute('list_files', {})).output).not.toContain('.coding-next');
      symlinkSync('tests/test_app.py', join(h.root, 'alias.py'));
      expect((await h.registry.execute('write_file', { path: 'alias.py', content: 'bad' })).success).toBe(false);
      rmSync(join(h.root, 'app.py'));
      symlinkSync('tests/test_app.py', join(h.root, 'app.py'));
      expect((await h.registry.execute('write_file', { path: 'app.py', content: 'bad' })).success).toBe(false);
      rmSync(join(h.root, 'app.py'));
      linkSync(join(h.root, 'tests/test_app.py'), join(h.root, 'app.py'));
      expect((await h.registry.execute('write_file', { path: 'app.py', content: 'bad' })).success).toBe(false);
      expect(readFileSync(join(h.root, 'tests/test_app.py'), 'utf8')).toBe('protected test');
    } finally { h.close(); }
  });
  it.each([[], ['app.py', 'app.py'], ['*.py'], ['/app.py'], ['src/../app.py'], ['src//app.py'], ['tests/'], ['.git/config'], ['.env'], ['app.py.coding-next'], ['app.py\n'], ['x'.repeat(241)], Array.from({ length: 33 }, (_, i) => `file${i}.py`)].map(paths => [paths]))('rejects malformed protected write policy %j', paths => {
    expect(() => harness(undefined, paths)).toThrow();
  });
  it('keeps extended deadlines coding-only and bounded', () => {
    expect(new WorkerPolicy({ ...data, max_job_seconds: 600 }).deadlineMs).toBe(600000);
    expect(() => new WorkerPolicy({ ...data, max_job_seconds: 601 })).toThrow();
    expect(() => new WorkerPolicy({ ...data, max_job_seconds: true })).toThrow();
    expect(() => new WorkerPolicy({ ...data, role: 'research', max_job_seconds: 181 })).toThrow();
    expect(new WorkerPolicy({ ...data, role: 'research', max_job_seconds: 180 }).deadlineMs).toBe(180000);
  });
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
