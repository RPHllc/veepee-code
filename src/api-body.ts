import type { IncomingMessage } from 'node:http';

export const WORKER_BODY_TIMEOUT_MS = 10_000;
export class BodyIntakeTimeoutError extends Error {}
const MAX_BODY_SIZE = 1024 * 1024; // upstream's existing character ceiling

/** Optional absolute intake deadline; ordinary upstream callers retain no deadline. */
export function readBody(req: IncomingMessage, timeoutMs?: number): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    let timer: ReturnType<typeof setTimeout> | undefined;
    const cleanup = () => {
      if (timer) clearTimeout(timer);
      req.off('data', onData); req.off('end', onEnd); req.off('error', onError); req.off('aborted', onAborted);
    };
    const onError = (error: Error) => { cleanup(); reject(error); };
    const onAborted = () => onError(new Error('Request body aborted'));
    const onEnd = () => { cleanup(); resolve(body); };
    const onData = (chunk: Buffer) => {
      body += chunk.toString();
      if (body.length > MAX_BODY_SIZE) {
        onError(new Error('Request body too large'));
        req.destroy();
      }
    };
    req.on('data', onData); req.on('end', onEnd); req.on('error', onError); req.on('aborted', onAborted);
    if (timeoutMs !== undefined) timer = setTimeout(() => {
      req.pause();
      onError(new BodyIntakeTimeoutError('Request body intake timed out'));
    }, timeoutMs);
  });
}
