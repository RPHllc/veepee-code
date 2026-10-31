# Research execution receipts v1

This optional API exists only when a server worker policy is present. It is a
bounded ephemeral executor interface for a supervising application, not a Vcode
scheduler or durable job database. The authenticated legacy API remains compatible.
Upstream tracking and MIT notices are unchanged.

All routes use the existing bearer authentication and ten-second worker body
intake limit. The server-controlled tool policy is enforced by the same registry
and Agent lock; clients cannot add tools or change models, policy or global settings.

- `GET /v1/research/status`: contract_version 1, per-process worker_instance UUID,
  busy, active_attempt_id, max_deadline_seconds, retention_seconds.
- `POST /v1/research/jobs`: strict version 1 / capability research.web request;
  UUID task_id, attempt_id, correlation_id, idempotency_key, expected_worker_instance;
  query 1–512 trimmed characters; max_results 1–3; deadline_seconds 1–120 and
  no greater than server policy. Returns 202 with immediate receipt, 200 for an
  identical retained request, 409 for busy/stale instance/identity conflict,
  422 invalid request/policy or 429 full receipt capacity. No queue or auto retry.
- `GET /v1/research/jobs/{attempt_id}`: snapshot or 404 absent/expired.
- `POST /v1/research/jobs/{attempt_id}/cancel`: strict worker_instance UUID and
  reason cancelled|timed_out. Returns actual state. Acknowledgment does not mean
  execution stopped. A terminal completed result wins over a later cancel.

A receipt echoes request identity and process instance. States: running,
cancellation_requested, completed, failed, timed_out, cancelled. `released=true`
is only set after the Agent generator **and underlying tool work** have settled.
The first stop reason wins. Supervisors must not interpret HTTP disconnect, 200,
missing receipt or an abort acknowledgment as completion/cancellation proof.
Process restart loses receipts: callers must reconcile, never blindly redispatch.

Receipts retain actual tool outcomes, structured search-source URLs, final/partial
output, sequenced bounded events and diagnostics. Model text/snippet URLs never
become source metadata. Tool arguments are bounded diagnostic strings, not executable
instructions. No reasoning text is retained. Model telemetry reports turn start,
first inference event latency, finish/interruption duration and available tokens;
first event can be a reasoning chunk, not visible text. Aggregate search duration
can overlap generation/parallel tool work.

Bounds: 32 in-memory receipts, 15 minutes after completion, 4000-byte final output,
six tool records (2048-byte outputs), ten 512-byte URLs, 64 events with bounded args.
Snapshot event eviction keeps JSON below 60,000 bytes and reports dropped_events.
The immutable research policy still governs total logical tool attempts and the
120-second ceiling. A job may use a smaller deadline; no shared model settings change.

Successful searches are reused only within the current acquired Agent run, keyed
by validated query + max_results default. Concurrent identical searches share one
promise. Failed searches are evicted; changed arguments and new jobs execute anew.
Cache hits still consume logical policy budget, while actual_tool_calls/cache_hits
separate inference/tool-loop repetition from network work. No cross-job cache.

Behavioral tests use mock inference and controlled tool work; full upstream tests
remain applicable. Local container/live DGX acceptance is recorded by the paired
Conductor PR. It does not authorize replacing the production Phase 5A1 image.
