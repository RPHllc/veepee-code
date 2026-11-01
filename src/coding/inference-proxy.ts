/** Fixed inference relay, inside a separate IPv4/IPv6 egress firewall namespace. */
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
const config = JSON.parse(readFileSync('/config/proxy.json', 'utf8')) as { base_url: string; model: string };
let calls = 0;
const server = createServer(async (req, res) => {
  const controller = new AbortController();
  const intake = setTimeout(() => { controller.abort(); req.destroy(); }, 5000);
  res.once('close', () => controller.abort());
  try {
    if (!((req.method === 'GET' && req.url === '/v1/models') || (req.method === 'POST' && req.url === '/v1/chat/completions'))) { res.writeHead(403).end(); return; }
    let body: string | undefined;
    if (req.method === 'POST') {
      if (++calls > 24) { res.writeHead(429).end(); return; }
      const chunks: Buffer[] = []; let bytes = 0;
      for await (const chunk of req) { bytes += chunk.length; if (bytes > 262144) throw new Error('intake'); chunks.push(chunk); }
      const data = JSON.parse(Buffer.concat(chunks).toString());
      if (data.model !== config.model || !Array.isArray(data.messages) || data.messages.length > 150 || (data.n !== undefined && data.n !== 1)) throw new Error('profile');
      data.max_tokens = Math.min(2048, Number.isInteger(data.max_tokens) && data.max_tokens > 0 ? data.max_tokens : 2048);
      body = JSON.stringify(data);
    }
    clearTimeout(intake);
    const upstream = await fetch(config.base_url.replace(/\/$/, '') + (req.method === 'GET' ? '/models' : '/chat/completions'), {
      method: req.method, body, headers: { 'Content-Type': 'application/json', 'Accept-Encoding': 'identity' }, redirect: 'error', signal: controller.signal,
    });
    if (!upstream.ok) { res.writeHead(502).end(); return; }
    res.setHeader('Content-Type', upstream.headers.get('content-type') ?? 'application/json');
    let bytes = 0;
    for await (const chunk of upstream.body!) {
      bytes += chunk.length; if (bytes > 4194304) throw new Error('response');
      if (!res.write(chunk)) await new Promise<void>(resolve => { res.once('drain', resolve); res.once('close', resolve); });
    }
    res.end();
  } catch { controller.abort(); if (!res.headersSent) res.writeHead(502); res.end(); }
  finally { clearTimeout(intake); }
});
server.requestTimeout = 5000;
server.listen(3030, '0.0.0.0');
