# Isolated headless worker

Upstream base: `vpontual/veepee-code@8383dc8400e1fae3bf89fb5b20528cabf0d96843`.
The MIT license and upstream CLI/TUI behavior remain intact. Keep the `upstream`
remote; update this base only through reviewed integration changes. No desktop
installation or global settings are used by this entry point.

`node dist/worker.js` requires `CONDUCTOR_WORKER_POLICY_PATH`,
`VCODE_WORKER_SETTINGS` and `VCODE_API_TOKEN_FILE`. Missing/malformed files,
unknown policy fields, unsupported tools, unsafe token permissions, short tokens,
invalid endpoints or bind conflicts fail startup. Only an audited `web_search`
implementation is eligible in this entry point. Other upstream profiles retain
all their existing capabilities and can use the reusable registry policy ceiling
with their own audited enrollment in subsequent work.

The server owns the policy. Registry execution checks its immutable allowlist
before argument parsing or calling a tool, including previously unknown names.
Agent per-request restrictions intersect this ceiling; omitted client tools means
only server-permitted tools, an empty list means no tools. Agent-local memory and
todo paths are subject to the same effective allowlist. Six permitted execution
attempts per job is a separate counter from the turn cap and wall-clock deadline.
The counter resets only when the trusted agent acquires its single-run lock.

This worker does not load MCP, remote tools, subagents, skills, LSP, desktop/project
settings, hooks, shell history, file mentions, image attachments, project context,
or persisted knowledge. Its upstream ContextManager uses an isolated prompt and
ephemeral knowledge state. The upstream agent loop and direct OpenAI adapter are
reused, including direct-backend compaction. No gateway or new model is installed.

All data/health routes require a 32+ character Bearer token. Direct execute,
Remote Connect and teacher side effects cannot be enabled with a worker policy.
The API model is pinned; each request starts a fresh conversation. Concurrent
requests receive 409. Jobs have a policy deadline, disconnect cancellation, a
bounded turn count and bounded search input/results. A deadline aborts inference
and closes the connection; an in-flight search can take up to its 15-second
transport ceiling to settle. Do not infer a resumable job or exactly-once protocol
from this API. A crash discards the job; Conductor integration/recovery is Phase
5A2. Nonstreaming inference failures return 502; streaming failures carry an error
event. Worker metadata reports actual tool result success independently of the
model's prose. Search results are untrusted and can be incomplete or incorrect.

The search tool uses a fixed operator endpoint, rejects redirects, validates
1–512 character queries and 1–10 results, streams at most 256 KiB, and bounds each
returned title/URL/snippet. It does not fetch result URLs. Arbitrary fetch/HTTP
and deep research remain unavailable pending separate SSRF review.

Build: `docker build --target runtime -f Dockerfile.worker .`.
Full suite: `docker build --target test -f Dockerfile.worker .`. The test stage
includes Git and pinned upstream history for history-based upstream tests, runs
unprivileged, and explicitly skips two tests requiring the upstream author's
absent private crosstown checkout (plus upstream platform sandbox skips). Policy
tests exercise registered denied-tool spies and the actual Agent/OpenAI streaming
adapter against an HTTP mock model, both chat API modes, auth, empty/omitted tool
lists, memory bypasses, input file expansion denial, boot/rebind, budgets and
hung-inference recovery. They are not live model/search evidence.

CI builds/tests on Linux amd64 and retains a Docker image archive for 14 days,
with source revision labels and an inspectable immutable image ID. It does not
publish or deploy. Operational Compose and acceptance evidence belong to the
canonical Conductor repository `deploy/vcode-research/`.

Reusable upstream contributions can be reviewed separately: registry policy and
job budget, isolated context/knowledge, API startup readiness/worker restrictions,
bounded SearXNG responses and portable private-checkout test skips. No upstream
PR or permission grant is implied by the downstream review.
