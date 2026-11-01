# Phase 5B-A review evidence

This is an isolated development implementation, not production acceptance. Research-worker configuration, deployed images, API contracts, Conductor and shared inference settings are unchanged. No coding deployment, merge, scheduler integration or production credentials are included.

## Reproducible checks

- `python3 scripts/coding/test_contract.py`: eight trusted-contract/control-plane tests, including uncertain Docker acknowledgments and failures after dispatch.
- `docker build --target test -f Dockerfile.worker .`: full existing Vcode suite, including coding adapter tests. Local Linux ARM64 run: 103 test files passed, one skipped; 1,455 tests passed, five skipped. Existing checkout/platform-dependent skips remain explicit.
- Build the `executor` and `agent` targets from `Dockerfile.coding`, then `python3 scripts/coding/test_linux.py`: 18 real Linux Docker tests with a deterministic synthetic inference fixture. These are actual kernel/container/process/filesystem/network tests, not mocked Docker calls. The separate contract suite does mock two exceptional Docker control-plane paths.
- The coding-worker GitHub workflow repeats the contract and container tests on native Linux amd64 and preserves development images, exact source labels, image IDs, archive checksum and per-case reports. The existing research-worker workflow continues running the complete Vcode suite. CI is development verification, never production acceptance.

The Linux suite covers single-file repair and multi-file changes with independently applied patches and external correctness oracles; actual failing/passing command exit codes; unapproved tools and commands; path traversal, initial and runtime symlinks; secrets and host paths; malicious repository instructions and Python user-site startup; readonly roots and IPC, non-root UID, dropped capabilities and seccomp; actual CPU/memory/PID cgroup settings, disk exhaustion and process-count limits; networkless executors and inference-only relay firewall; deadline/cancellation with detached subprocess trees; launcher interruption/recovery; agent container interruption; and controlled Python compilation. Memory limits are inspected at the kernel boundary; this suite does not deliberately OOM the Docker host. No production-host fault injection is performed.

Set `CODING_EVIDENCE_DIR` to preserve per-case `result.json`, command records, patch, integrity manifest and synthetic base Git bundle. Replay the patch against the recorded bundle/base, check its SHA-256 and run an independent oracle. A model summary is not correctness evidence. Intermediate command failures remain recorded even when final validation succeeds.

## Limited live-model evaluation

`scripts/coding/evaluate.py` is an explicit opt-in, two-task evaluation against an operator-enrolled numeric inference endpoint. It uses only disposable synthetic repositories, the same sandbox and command process evidence, and independent patch-application oracles. It does not install dependencies, change inference settings, access production checkouts or invoke research services. Functional correctness and observed wall time are recorded separately. The two-task evaluation used source `2e0f0283aa5ea4d614beabbcc28552283e87481a` on actual Linux ARM64 Docker Desktop, against the live DGX `Qwen/Qwen3.6-35B-A3B-FP8` advertised alias (model root `nvidia/Qwen3.6-35B-A3B-NVFP4`). Both completed, independently applied patches and passed external correctness oracles. No shared configuration was changed.

| Task | Functional correctness | Observed wall seconds | Worker milliseconds | Model turns | Actual tools | Generation milliseconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single-file addition repair | Passed | 17.882 | 17757 | 5 | 5 | 12185 |
| Two-file addition/doubling repair | Passed | 17.736 | 17569 | 6 | 9 | 12054 |

Each recorded four actual exit-zero command runs: model-requested unittest/syntax plus independent launcher final unittest/syntax validation. First SSE-event observations were 201–326 ms and 203–378 ms respectively; these are first protocol events, not a claim of time to first visible text token. This small synthetic evaluation demonstrates functionality, not general coding accuracy or production acceptance.

Live-tested ARM64 immutable image IDs:

- Agent: `sha256:e8c603accc23954044e76190c0de5520492886c786975737507d45f6e4f60ebb`.
- Executor: `sha256:9850588504ac271ffd3f7f19c6447e80de2f3303a88ea41be07fa2a6662e3840`.

Owner-only evidence is retained under `~/.local/share/conductor/diagnostics/phase5b-a/live-dgx-verified/` (`summary.json`, both reports, per-task patch/commands/integrity manifest and synthetic base bundle). The final local 18-case suite passed in 91.592 seconds; its retained evidence is under `phase5b-a/linux-final/`. Final PR-head amd64 image identities and CI links are recorded in the PR and its downloaded source manifest, distinguishing them from these live-tested ARM64 images. The initial two-task run also passed, but its evidence copier omitted a launch manifest and reused a directory across cases. The corrected harness repeated both tasks to produce the complete, distinct manifests cited here, whose every recorded artifact hash was independently checked. A regression checks copied manifests during CI. Replay of an existing live attempt reports unknown stop state rather than claiming the original job stopped. Final PR-head changes after this checkpoint are evidence/test-only; CI rebuilds exact-head images.

## Review and activation boundary

The upstream base remains `8383dc8400e1fae3bf89fb5b20528cabf0d96843`; MIT notices remain intact. Shared Agent changes are optional isolated prompts, suppression of automatic filesystem readers for policy profiles, and trusted test-verification metadata. Ordinary upstream behavior and the research default prompt are retained and covered by the full suite.

Limits: the Docker engine, Linux kernel, protected operator profile, trusted launcher and pinned images form the boundary. This is not kernel-exploit isolation or a general-purpose arbitrary repository executor. Initial raw-tree limits are 512 files, 256 KiB per file and 2 MiB total; links, submodules and common secret paths are rejected. Only enrolled Python unittest, Node test and Python syntax profiles exist. Dependency installation is controlled image build-time work. CPU, memory and PID budgets are per compartment, with conservative aggregate maxima reported. No persistent Vcode job scheduler/database or coding HTTP listener is introduced. Recovery may honestly report lost artifacts or uncertain control-plane cleanup; it never synthesizes successful tests or blindly retries.

Review [the contract](coding-job-v1.md) and [sandbox design](coding-worker.md), including the proposed future Phase 5B-B Conductor interface. Production activation, authorization integration and any additional command/dependency profiles require separate review. This phase stops at an unmerged reviewable Vcode PR.
