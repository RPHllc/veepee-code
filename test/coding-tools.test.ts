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
  it('allows enrolled writes and bounded reads of protected files', async () => {
    const h = harness(undefined, ['app.py', 'src/new.py']);
    try {
      mkdirSync(join(h.root, 'tests'));
      writeFileSync(join(h.root, 'README.md'), 'protected requirements');
      writeFileSync(join(h.root, 'tests/test_app.py'), 'protected test');
      writeFileSync(join(h.root, 'app.py'), 'return a - b');
      expect((await h.registry.execute('read_file', { path: 'README.md' })).output).toBe('protected requirements');
      expect((await h.registry.execute('read_file', { path: 'tests/test_app.py' })).output).toBe('protected test');
      expect((await h.registry.execute('edit_file', { path: 'app.py', old_string: 'a - b', new_string: 'a + b' })).success).toBe(true);
      expect((await h.registry.execute('write_file', { path: 'src/new.py', content: 'allowed source' })).success).toBe(true);
      expect(readFileSync(join(h.root, 'app.py'), 'utf8')).toBe('return a + b');
      expect((await h.registry.execute('list_files', {})).output).not.toContain('.coding-next');
    } finally { h.close(); }
  });
  it.each(['README.md', 'tests/test_app.py', 'app.py.bak', 'app.py.coding-next', './app.py', 'src/../app.py'])('terminally denies unenrolled mutation %s before matching', async path => {
    const h = harness(undefined, ['app.py']);
    try {
      mkdirSync(join(h.root, 'tests'));
      writeFileSync(join(h.root, 'README.md'), 'protected');
      writeFileSync(join(h.root, 'tests/test_app.py'), 'protected');
      writeFileSync(join(h.root, 'app.py'), 'original');
      const denied = h.registry.execute('edit_file', { path, old_string: 'absent', new_string: 'bad' });
      const queued = h.registry.execute('write_file', { path: 'app.py', content: 'must not run' });
      expect((await denied).error).toBe('Coding workspace policy denied');
      expect((await queued).error).toBe('Coding workspace policy denied');
      expect(readFileSync(join(h.root, 'app.py'), 'utf8')).toBe('original');
      expect(readFileSync(join(h.root, 'tests/test_app.py'), 'utf8')).toBe('protected');
    } finally { h.close(); }
  });
  it.each(['symlink', 'hardlink'])('denies enrolled %s mutation', async kind => {
    const h = harness(undefined, ['app.py']);
    try {
      writeFileSync(join(h.root, 'protected.py'), 'protected');
      (kind === 'symlink' ? symlinkSync : linkSync)(join(h.root, 'protected.py'), join(h.root, 'app.py'));
      expect((await h.registry.execute('write_file', { path: 'app.py', content: 'bad' })).error).toBe('Coding workspace policy denied');
      expect(readFileSync(join(h.root, 'protected.py'), 'utf8')).toBe('protected');
    } finally { h.close(); }
  });
  it('recovers from missing files and ambiguous edits without granting extra authority', async () => {
    const h = harness(undefined, ['app.py']);
    try {
      expect((await h.registry.execute('read_file', { path: 'missing.py' })).error).toContain('does not exist');
      writeFileSync(join(h.root, 'app.py'), 'same same');
      expect((await h.registry.execute('edit_file', { path: 'app.py', old_string: 'same', new_string: 'bad' })).error).toBe('Unique exact match required');
      expect((await h.registry.execute('edit_file', { path: 'app.py', old_string: 'absent', new_string: 'bad' })).error).toBe('Unique exact match required');
      expect((await h.registry.execute('edit_file', { path: 'app.py', old_string: 'same same', new_string: 'correct' })).success).toBe(true);
      expect(readFileSync(join(h.root, 'app.py'), 'utf8')).toBe('correct');
    } finally { h.close(); }
  });
  it('sanitizes unexpected failures and stops queued workspace operations', async () => {
    const h = harness(async () => { throw new Error('private host detail'); });
    try {
      expect((await h.registry.execute('run_command', { command_id: 'python_unittest' })).error).toBe('Coding workspace internal failure');
      expect((await h.registry.execute('write_file', { path: 'new.py', content: 'bad' })).error).toBe('Coding workspace internal failure');
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
