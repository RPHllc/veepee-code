// Deterministic test-only OpenAI server. No external network, credentials or model.
import { createServer } from 'node:http';
createServer(async (req, res) => {
  if (req.url === '/v1/models') { res.end(JSON.stringify({ data: [{ id: 'synthetic-coder' }] })); return; }
  let body = ''; for await (const part of req) body += part;
  const data = JSON.parse(body);
  const text = data.messages.find(m => m.role === 'user')?.content ?? '';
  const step = data.messages.filter(m => m.role === 'tool').length;
  let calls = [];
  const call = (name, args) => calls.push({ index: calls.length, id: 'tool_' + step + '_' + calls.length, type: 'function', function: { name, arguments: JSON.stringify(args) } });
  if (text.includes('CASE:hang')) return;
  if (text.includes('CASE:repair')) {
    if (step === 0) call('run_command', { command_id: 'python_unittest' });
    if (step === 1) call('read_file', { path: 'app.py' });
    if (step === 2) call('edit_file', { path: 'app.py', old_string: 'return a - b', new_string: 'return a + b' });
    if (step === 3) call('run_command', { command_id: 'python_unittest' });
  } else if (text.includes('CASE:multi')) {
    if (step === 0) { call('edit_file', { path: 'app.py', old_string: 'return a - b', new_string: 'return a + b' }); call('edit_file', { path: 'helper.py', old_string: 'return x', new_string: 'return x * 2' }); }
    if (step === 2) call('run_command', { command_id: 'python_unittest' });
  } else if (text.includes('CASE:denied:') && step === 0) call(text.split('CASE:denied:')[1].split(' ')[0], { command: 'git push; cat /etc/shadow' });
  else if (text.includes('CASE:escape') && step === 0) call('write_file', { path: '../escape.py', content: 'escaped' });
  else if (text.includes('CASE:secret') && step === 0) call('read_file', { path: '/config/job.json' });
  else if (text.includes('CASE:command-denied') && step === 0) call('run_command', { command_id: 'bash' });
  else if (text.includes('CASE:symlink')) {
    if (step === 0) call('run_command', { command_id: 'python_unittest' });
    if (step === 1) call('write_file', { path: 'link/exploit.py', content: 'escaped' });
  } else if (text.includes('CASE:probe') || text.includes('CASE:tree')) {
    if (step === 0) call('run_command', { command_id: 'python_unittest' });
  }
  if (text.includes('CASE:tree') && step >= 1) return;
  if (data.messages[0]?.content?.includes('REPO_AUTH_OVERRIDE')) { calls = []; call('bash', {command:'not authorized'}); }
  res.setHeader('Content-Type', 'text/event-stream');
  const delta = calls.length ? { tool_calls: calls } : { content: 'All tests passed. Work complete.' };
  res.end('data: ' + JSON.stringify({ choices: [{ delta, finish_reason: calls.length ? 'tool_calls' : 'stop' }] }) + '\n\ndata: [DONE]\n\n');
}).listen(8000, '0.0.0.0');
