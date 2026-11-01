# code.patch v1 and proposed Conductor interface

This is a separate disposable coding capability. It never grants GitHub mutation,
PR creation, merge, deployment, host administration or research-worker permissions.
Conductor remains the sole future durable scheduler and authorization authority.

## Protected operator profile

The launcher requires an owner-only regular JSON file (0600), not a job-selected
configuration. Its artifact directory must already exist, owner-only0700. Images
must be exact immutable `sha256:` IDs with the coding-agent/executor labels and
Linux platform. The examples contain **nonusable placeholder identities**; replace
them only with independently inspected development builds and disposable input.
No credential is created or read from production by these examples.

```json
{
  "contract_version": 1,
  "agent_image": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "executor_image": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "docker_context": "desktop-linux",
  "inference": {"base_url": "http://10.0.154.246:8000/v1", "model": "Qwen/Qwen3.6-35B-A3B-FP8"},
  "inference_network": null,
  "repositories": {
    "fixture": {"path": "/protected/disposable-fixture", "base_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "commands": ["python_unittest", "python_syntax"]}
  },
  "artifact_directory": "/protected/coding-artifacts"
}
```

A null inference network creates a per-job egress bridge; a named preapproved
bridge is useful for a synthetic model container. The separate namespace firewall
still permits only the enrolled numeric inference IP/port. No DNS, discovery,
redirect, arbitrary URL, shell string, model-selected command or package install.
Repository path is a trusted **local object source**, never mounted into a job.
The raw tree is copied to a fresh tmpfs checkout. Reject symlinks/submodules,
secret-path names, unsupported filenames and oversized trees before executing.

## Job input

[Machine-readable request schema](coding-job-v1.schema.json). Unknown/duplicate
fields, unsupported versions, noncanonical IDs, unenrolled repository/revision,
excessive instructions or expanded budgets fail closed before dispatch.

```json
{
  "contract_version": 1,
  "capability": "code.patch",
  "task_id": "10000000-0000-4000-8000-000000000001",
  "attempt_id": "10000000-0000-4000-8000-000000000002",
  "correlation_id": "10000000-0000-4000-8000-000000000003",
  "idempotency_key": "10000000-0000-4000-8000-000000000004",
  "repository": "fixture",
  "base_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "instructions": "Repair the failing addition test without changing test expectations.",
  "budgets": {"wall_seconds": 120, "command_seconds": 15, "cpus": 0.5, "memory_mb": 256, "workspace_mb": 16, "pids": 64, "tool_calls": 32}
}
```

The four UUIDs bind supervisory task, unique execution attempt, trace and submission
identity. Reusing an attempt directory never dispatches again. The launcher has
no cross-attempt idempotency database; Conductor must not create another attempt
merely because transport/process supervision was lost. Budget ceilings and aggregate
resource semantics are in [sandbox documentation](coding-worker.md#local-use-and-resources).
The input JSON is bounded to16KiB, instructions to8000 UTF-8 bytes.

## Job output and integrity

The foreground CLI emits one JSON result, also durably stored in the protected
attempt artifact directory. Required bindings: contract_version, capability,
task_id, attempt_id, correlation_id, idempotency_key, repository and exact base_sha.
The result additionally binds the protected profile's canonical SHA256 and model ID.

| Field | Meaning |
| --- | --- |
| state | completed, failed, timed_out or cancelled; not inferred from model prose/exit0 alone |
| termination_reason | Closed worker/control failure class; cleanup_unconfirmed means execution stop is uncertain |
| execution_stopped | Independently confirmed absence of attempt containers, not HTTP disconnect/parent exit |
| elapsed_ms | Monotonic total observation including teardown; null if interruption destroyed timing |
| changed_files | Trusted final inventory: added/modified/deleted, mode, bytes and SHA256; null when lost |
| patch | Actual binary-safe Git diff against base, artifact path/bytes/SHA256/status; null after unrecoverable interruption |
| commands | Actual fixed argv, observed exit code (or unknown/null), bounded stdout/base64/hash/stderr, truncation, elapsed time, OOM observation, test-tree hash and executor-stop proof |
| tests_not_executed | Enrolled final validation profiles without a recorded attempt; commands with interrupted/unknown exits are never passing evidence |
| command_history_status | recorded, or interrupted with whatever durable command evidence survived |
| artifact_hashes | Hashes of patch and captured command artifacts; artifacts.json also hashes result and launch manifest |
| images | Exact immutable agent/executor identities, observed architecture/source labels where available |
| network_evidence / resource_limits | Actual installed kernel firewall rules and conservative workload resource bounds |
| agent | Bounded model summary/events/diagnostics, explicitly not authoritative test or authorization evidence |

Only completed state, matching identities/profile/image pins, complete verified
patch artifacts, independently stopped execution and passing **final** validation
can support successful supervisory completion. An earlier failed test followed
by actual repair is retained. A model saying “all tests passed” with no executed
validation is insufficient. All command records are capped (8 runs,32KiB stdout,
4KiB stderr each); output_base64 preserves bytes for output_sha256 verification.
Workspace-file and patch hashes are integrity checks, not proof of code correctness
or a cryptographic attestation against a compromised host/operator/kernel.

Before returning completion, final validation runs after the model exits, and
the exact source tree must remain unchanged during/after validation. The test
framework and repository tests can themselves lie; an external acceptance oracle
must apply the patch to the recorded base and check functional expectations.
These oracles exist for both synthetic acceptance tasks. No patch is applied to
an owner/production checkout by the coding worker.

Exit0 accompanies completed; exit1 accompanies a structured unsuccessful job.
Exit2 indicates input/control/report failure. A definite preflight failure says
execution_started=false and dispatch_may_have_started=false. If dispatch may
have started, execution_started/execution_stopped are unknown rather than a false
preflight claim. Missing/malformed results after process loss need reconciliation.
Do not release an uncertainty fence merely because the CLI exited.

## Cancellation and recovery

SIGTERM is cancellation, not a guarantee. The launcher first stops all mutators,
then collects available partial evidence read-only, and verifies teardown before
reporting cancelled. Each container has a separate bounded lifetime; a passive
holder has a short additional artifact grace, not repository execution authority.
Lost create acknowledgments remain uncertain until the exact named container is
observed and removed, or a subsequent recovery establishes stop. Docker-unavailable
cleanup reports failed/cleanup_unconfirmed, not cancellation success.

`recover --attempt` is one-shot reconciliation, not rescheduling. It binds the
original protected profile/manifest, removes only that attempt's namespace and
reports failed/interrupted with no success claim. In-progress command exit codes
remain unknown. If an earlier final result exists it is returned idempotently;
recovery does not fabricate a new pass or rerun tests. Missing repository object
storage need not prevent cleanup. Preserve artifacts and original base objects
until Conductor has independently accepted/rejected them.

## Proposed Phase 5B-B interface

**Proposal only; no Conductor implementation or production enrollment in 5B-A.**

1. Conductor validates/authenticates an enrolled code.patch request and persists
   its task/attempt/correlation/idempotency, immutable repository revision, policy
   and resource/image fingerprint before starting execution. Reserve the aggregate
   resource ceiling and the appropriate per-capability concurrency slot transactionally.
2. A separately provisioned coding-executor account runs this fixed one-shot
   launcher through a constrained local systemd unit/OS boundary. Conductor must
   **not** gain Docker group/socket/root, nor accept client-selected ExecStart,
   mounts, model endpoints or image references. A remote transport would require
   authenticated/encrypted versioned messages and separate enrollment review.
3. Stream bounded progress/process metadata to the existing Conductor task
   supervisor. Keep the worker ephemeral. Native SQLite remains the sole durable
   lifecycle and outbox. Do not create another scheduler, agent architecture,
   authorization service or LLM gateway.
4. Cancellation requests signal the exact owned attempt, wait for stopped-execution
   evidence and preserve partial artifacts. After restart or uncertain transport,
   reconcile the original attempt; never duplicate dispatch or overwrite a newer
   attempt with a late result. Missing stop proof remains unresolved.
5. Verify bindings, source/image identities, result schema, artifact hashes,
   changed-file inventory and independently applicable patch. Optionally rerun
   independent validation in another isolated checkout before task acceptance.
   Treat model narratives and repository instructions as untrusted.
6. Patch acceptance is distinct from permission to push, open a PR, merge or deploy.
   Those require later typed capabilities, credentials and Conductor/owner authority.
   Keep coding transcripts/artifacts out of generic task listings and public channels.
   Add protected artifact retention/backup under Conductor control without deleting
   unresolved evidence or idempotency fences to satisfy a quota.

The current launcher authenticates a local operator through OS/file ownership;
it is not yet an enrolled Conductor worker API. 5B-B must provide the fixed
privilege boundary and durable supervision before any automated production use.
