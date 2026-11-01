#!/usr/bin/env python3
"""One-shot operator coding executor. No scheduler/database, host shell or job Docker socket.

run: protected profile + strict request -> isolated Agent -> actual executor records -> patch.
recover: prove stopped execution after interruption; never infer successful completion or retry.
"""

import argparse
import base64
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit
from contract import (
    COMMANDS,
    Rejected,
    canonical,
    identifier,
    load_profile,
    read_json,
    request,
    sha,
)
from git_snapshot import GitSnapshot


DISPATCH_MAY_HAVE_STARTED = False


class Stopped(Exception):
    pass


class Launcher:
    def __init__(self, profile, job):
        self.profile, self.job = profile, job
        self.b = job["budgets"]
        self.started = time.monotonic()
        self.deadline = self.started + self.b["wall_seconds"]
        self.stop_reason = None
        self.uncertain_creations = set()
        self.resource_creation_uncertain = False
        self.capture_started = False
        self.names = []
        self.networks = []
        self.volume = None
        self.child = None
        self.prefix = "vcode-coding-" + job["attempt_id"]
        self.records = []
        self.git = None
        self.agent = {}
        self.sequence = 0
        self.workspace = None
        self.collecting = False
        self.holder = None
        self.artifacts = Path(profile["artifact_directory"]) / job["attempt_id"]
        self.artifacts.mkdir(
            mode=0o700
        )  # Existing attempt is a fence, never overwritten/retried.
        self.write(
            "launch.json",
            {
                "request": job,
                "profile_sha256": sha(canonical(profile)),
                "source_images": [profile["agent_image"], profile["executor_image"]],
            },
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.cancel)

    def cancel(self, *_):
        self.stop_reason = "cancelled"

    def remaining(self):
        if self.stop_reason:
            raise Stopped()
        if time.monotonic() >= self.deadline:
            self.stop_reason = "timed_out"
            raise Stopped()
        return max(1, math.ceil(self.deadline - time.monotonic()))

    def write(self, name, value):
        data = canonical(value) if not isinstance(value, bytes) else value
        path = self.artifacts / name
        temp = path.with_suffix(path.suffix + ".next")
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temp.chmod(0o600)
        temp.replace(path)
        return sha(data)

    def capture(
        self,
        args,
        data=None,
        limit=1048576,
        env=None,
        timeout=None,
        truncate=False,
        cleanup=False,
    ):
        self.capture_started = False
        if not cleanup:
            if self.collecting:
                if time.monotonic() > self.collection_deadline:
                    raise TimeoutError("artifact_collection_budget")
            else:
                self.remaining()
        budget = timeout or (
            10 if cleanup or self.collecting else min(10, self.remaining())
        )
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self.capture_started = True
        if data is not None:
            # Inputs are bounded Git blobs/config; Git consumes before producing output.
            process.stdin.write(data)
            process.stdin.close()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "out")
        selector.register(process.stderr, selectors.EVENT_READ, "err")
        output = bytearray()
        errors = bytearray()
        truncated = False
        end = time.monotonic() + budget
        try:
            while selector.get_map():
                if not cleanup and not self.collecting:
                    self.remaining()
                if (
                    self.collecting
                    and not cleanup
                    and time.monotonic() > self.collection_deadline
                ):
                    raise TimeoutError("artifact_collection_budget")
                if time.monotonic() >= end:
                    raise TimeoutError("process_budget")
                for key, _ in selector.select(0.05):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target = output if key.data == "out" else errors
                    cap = limit if key.data == "out" else 4096
                    room = max(0, cap - len(target))
                    target.extend(chunk[:room])
                    if len(chunk) > room:
                        truncated = True
                        if key.data == "out" and not truncate:
                            raise Rejected("bounded_output_exceeded")
            process.wait(timeout=1)
            return (
                process.returncode,
                bytes(output),
                {"truncated": truncated, "stderr": errors.decode(errors="replace")},
            )
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)

    def docker(self, args, **kwargs):
        try:
            return self.capture(
                ["docker", "--context", self.profile["docker_context"], *args], **kwargs
            )
        except (Stopped, TimeoutError):
            if self.capture_started:
                if args[0] in ("run", "create") and "--name" in args:
                    self.uncertain_creations.add(args[args.index("--name") + 1])
                if args[:2] in (["network", "create"], ["volume", "create"]):
                    self.resource_creation_uncertain = True
            raise

    def checked(self, args, **kwargs):
        code, output, meta = self.docker(args, **kwargs)
        if code:
            raise Rejected("docker_operation_failed:" + meta["stderr"][:200])
        return output

    def flags(self, name, image=None, network="none", memory=None):
        self.names.append(name)
        return [
            "--name",
            name,
            "--label",
            "vcode.coding.attempt=" + self.job["attempt_id"],
            "--label",
            "vcode.coding.profile=" + sha(canonical(self.profile)),
            "--read-only",
            "--ipc",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--init",
            "--user",
            "10001:10001",
            "--network",
            network,
            "--cpus",
            str(self.b["cpus"]),
            "--memory",
            str(memory or self.b["memory_mb"]) + "m",
            "--memory-swap",
            str(memory or self.b["memory_mb"]) + "m",
            "--pids-limit",
            str(self.b["pids"]),
            "--log-driver",
            "none",
            "--ulimit",
            "nofile=128:128",
            "--ulimit",
            "core=0",
            "--env",
            "HOME=/workspace",
            "--env",
            "LC_ALL=C.UTF-8",
            "--workdir",
            "/workspace",
        ]

    def mount(self, writable=True):
        return [
            "--mount",
            "type=volume,src="
            + self.volume
            + ",dst=/workspace"
            + ("" if writable else ",readonly"),
        ]

    def image(self, image, role):
        value = json.loads(self.checked(["image", "inspect", image]))[0]
        if (
            value["Id"] != image
            or value["Os"] != "linux"
            or value["Config"].get("Labels", {}).get("org.rphllc.profile") != role
        ):
            raise Rejected("image_identity_or_profile")
        return {
            "id": value["Id"],
            "architecture": value["Architecture"],
            "source_sha": value["Config"]["Labels"].get(
                "org.opencontainers.image.revision"
            ),
        }

    def snapshot(self):
        self.sequence += 1
        name = self.prefix + "-snapshot-" + str(self.sequence)
        output = self.checked(
            [
                "run",
                "--rm",
                *self.flags(name),
                *self.mount(False),
                self.profile["executor_image"],
                "timeout",
                "--signal=KILL",
                str(10 if self.collecting else min(10, self.remaining())),
                "/usr/bin/python3",
                "-I",
                "-B",
                "/opt/coding/snapshot.py",
                "collect",
            ],
            limit=4000000,
        )
        return json.loads(output)

    def diff(self):
        return self.git.diff(self.snapshot())

    def command(self, command_id, validation=False):
        repo = self.profile["repositories"][self.job["repository"]]
        if len(self.records) >= 8:
            raise Rejected("command_count_budget")
        if command_id not in repo["commands"] or command_id not in COMMANDS:
            raise Rejected("command_denied")
        self.sequence += 1
        name = self.prefix + "-exec-" + str(self.sequence)
        seconds = min(self.b["command_seconds"], self.remaining())
        start = time.monotonic()
        before = sha(canonical(self.snapshot()))
        record = {
            "command_id": command_id,
            "argv": COMMANDS[command_id],
            "exit_code": None,
            "elapsed_ms": 0,
            "output": "",
            "output_base64": "",
            "stderr": "",
            "output_sha256": sha(b""),
            "truncated": False,
            "executor_stopped": False,
            "validation": validation,
            "input_tree_sha256": before,
            "termination_reason": "in_progress",
            "oom_killed": None,
        }
        self.records.append(record)
        self.write("commands.json", self.records)
        try:
            self.checked(
                [
                    "create",
                    *self.flags(name),
                    *self.mount(),
                    self.profile["executor_image"],
                    "timeout",
                    "--signal=KILL",
                    str(seconds),
                    *COMMANDS[command_id],
                ]
            )
            code, output, meta = self.docker(
                ["start", "--attach", name],
                limit=32768,
                truncate=True,
                timeout=seconds + 3,
            )
            state = json.loads(self.checked(["container", "inspect", name]))[0]["State"]
            if state["Running"] or state["StartedAt"].startswith("0001-"):
                raise Rejected("executor_state_unconfirmed")
            code = state["ExitCode"]
            record["oom_killed"] = state["OOMKilled"]
            self.remove(name)
            stopped = self.absent(name)
            if not stopped:
                raise Rejected("executor_stop_unconfirmed")
        except TimeoutError:
            self.remove(name)
            code, output, meta = None, b"", {"truncated": False, "stderr": ""}
            stopped = self.absent(name)
        record.update(
            exit_code=code,
            elapsed_ms=round((time.monotonic() - start) * 1000),
            output=output.decode(errors="replace"),
            output_base64=base64.b64encode(output).decode(),
            stderr=meta["stderr"],
            output_sha256=sha(output),
            truncated=meta["truncated"],
            executor_stopped=stopped,
            termination_reason="exited"
            if code == 0
            else "oom_killed"
            if record["oom_killed"]
            else "timeout_or_kill"
            if code in (None, 137)
            else "nonzero_exit",
        )
        self.write("commands.json", self.records)
        return {
            "success": code == 0 and stopped,
            "output": json.dumps(record),
            **(
                {}
                if code == 0
                else {
                    "error": json.dumps(
                        {
                            "exit_code": code,
                            "stdout": record["output"],
                            "stderr": record["stderr"],
                        }
                    )[:8000]
                }
            ),
        }

    def absent(self, name):
        code, _, meta = self.docker(["container", "inspect", name], cleanup=True)
        if not code:
            return False
        if "No such" in meta["stderr"]:
            return True
        raise Rejected("container_state_unknown")

    def remove(self, name):
        code, _, meta = self.docker(["rm", "-f", name], cleanup=True)
        if code and "No such" not in meta["stderr"]:
            raise Rejected("container_cleanup_unconfirmed")

    def setup(self, directory):
        self.images = {
            "agent": self.image(self.profile["agent_image"], "coding-agent"),
            "executor": self.image(self.profile["executor_image"], "coding-executor"),
        }
        info = json.loads(self.checked(["info", "--format", "{{json .}}"]))
        if info["OSType"] != "linux" or not any(
            "seccomp" in x for x in info["SecurityOptions"]
        ):
            raise Rejected("linux_seccomp_required")
        repo = self.profile["repositories"][self.job["repository"]]
        self.git = GitSnapshot(
            self.capture, repo["path"], self.job["base_sha"], Path(directory) / "git"
        )
        self.volume = self.prefix + "-workspace"
        self.checked(
            [
                "volume",
                "create",
                "--label",
                "vcode.coding.attempt=" + self.job["attempt_id"],
                "--driver",
                "local",
                "--opt",
                "type=tmpfs",
                "--opt",
                "device=tmpfs",
                "--opt",
                f"o=size={self.b['workspace_mb']}m,uid=10001,gid=10001,mode=0700",
                self.volume,
            ]
        )
        private, egress = (
            self.prefix + "-private",
            self.profile["inference_network"] or self.prefix + "-inference",
        )
        networks = [(private, True)] + (
            [(egress, False)] if self.profile["inference_network"] is None else []
        )
        if self.profile["inference_network"] is not None:
            existing = json.loads(self.checked(["network", "inspect", egress]))[0]
            if existing["Driver"] != "bridge":
                raise Rejected("bridge_inference_network_required")
        for name, internal in networks:
            self.networks.append(name)
            self.checked(
                [
                    "network",
                    "create",
                    *(["--internal"] if internal else []),
                    "--label",
                    "vcode.coding.attempt=" + self.job["attempt_id"],
                    name,
                ]
            )
        subnet = json.loads(self.checked(["network", "inspect", private]))[0]["IPAM"][
            "Config"
        ][0]["Subnet"]
        u = urlsplit(self.profile["inference"]["base_url"])
        guard = self.prefix + "-guard"
        self.names.append(guard)
        self.checked(
            [
                "run",
                "-d",
                "--name",
                guard,
                "--label",
                "vcode.coding.attempt=" + self.job["attempt_id"],
                "--network",
                egress,
                "--read-only",
                "--ipc",
                "none",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "NET_ADMIN",
                "--security-opt",
                "no-new-privileges=true",
                "--memory",
                "64m",
                "--memory-swap",
                "64m",
                "--cpus",
                ".1",
                "--pids-limit",
                "16",
                "--log-driver",
                "none",
                "--user",
                "0:0",
                self.profile["executor_image"],
                "/usr/bin/python3",
                "-I",
                "-B",
                "/opt/coding/netguard.py",
                u.hostname,
                str(u.port),
                subnet,
                str(self.remaining()),
            ]
        )
        self.checked(["network", "connect", private, guard])
        # Read the actual installed kernel rules. Failure to install either
        # IPv4 or IPv6 deny policy makes the profile unenforceable.
        for _ in range(40):
            self.remaining()
            code, rules, _ = self.docker(["exec", guard, "/usr/sbin/iptables", "-S"])
            if not code and b"-P OUTPUT DROP" in rules and b"--dport 3030" in rules:
                break
            time.sleep(0.05)
        else:
            raise Rejected("egress_guard_unavailable")
        v6 = self.checked(["exec", guard, "/usr/sbin/ip6tables", "-S"])
        assert b"-P OUTPUT DROP" in v6
        self.network_evidence = {
            "ipv4_rules": rules.decode(),
            "ipv6_rules": v6.decode(),
            "executor_network": "none",
            "agent_network_internal": True,
        }
        address = json.loads(self.checked(["inspect", guard]))[0]["NetworkSettings"][
            "Networks"
        ][private]["IPAddress"]
        config = Path(directory) / "config"
        config.mkdir(mode=0o755)
        for name, value in [
            ("base.json", self.git.base),
            ("proxy.json", self.profile["inference"]),
            (
                "job.json",
                {
                    "instructions": self.job["instructions"],
                    "model": self.profile["inference"]["model"],
                    "inference_url": "http://" + address + ":3030/v1",
                    "commands": repo["commands"],
                    "max_tool_calls": self.b["tool_calls"],
                    "max_seconds": self.remaining(),
                },
            ),
        ]:
            path = config / name
            path.write_bytes(canonical(value))
            path.chmod(0o444)
        # Docker Desktop cannot traverse a private parent via bind mounts when
        # resolving a source directory. Mount only generated non-secret job data.
        self.config_mount = [
            "--mount",
            "type=bind,src=" + str(config) + ",dst=/config,readonly",
        ]
        # Keep the size-limited tmpfs mounted between short-lived executors.
        # This holder runs only trusted sleep, with a bounded artifact grace period.
        self.holder = self.prefix + "-workspace-lease"
        self.checked(
            [
                "run",
                "-d",
                *self.flags(self.holder),
                *self.mount(),
                self.profile["executor_image"],
                "timeout",
                "--signal=KILL",
                str(self.remaining() + 20),
                "sleep",
                "201",
            ]
        )
        self.sequence += 1
        name = self.prefix + "-init"
        self.checked(
            [
                "run",
                "--rm",
                *self.flags(name),
                *self.mount(),
                *self.config_mount,
                self.profile["executor_image"],
                "timeout",
                "--signal=KILL",
                str(min(10, self.remaining())),
                "/usr/bin/python3",
                "-I",
                "-B",
                "/opt/coding/snapshot.py",
                "init",
            ]
        )
        proxy = self.prefix + "-proxy"
        self.checked(
            [
                "run",
                "-d",
                *self.flags(proxy, network="container:" + guard),
                *self.config_mount,
                self.profile["agent_image"],
                "timeout",
                "--signal=KILL",
                str(self.remaining()),
                "node",
                "/app/dist/coding/inference-proxy.js",
            ]
        )
        return private

    def run_agent(self, network):
        name = self.prefix + "-agent"
        self.checked(
            [
                "create",
                "-i",
                *self.flags(name, network=network),
                *self.mount(),
                *self.config_mount,
                self.profile["agent_image"],
                "timeout",
                "--signal=KILL",
                str(self.remaining()),
                "node",
                "/app/dist/coding-worker.js",
            ]
        )
        cmd = [
            "docker",
            "--context",
            self.profile["docker_context"],
            "start",
            "--attach",
            "--interactive",
            name,
        ]
        child = self.child = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        seen = set()
        try:
            while True:
                self.remaining()
                if not selector.select(0.05):
                    if child.poll() is not None:
                        break
                    continue
                chunk = os.read(child.stdout.fileno(), 65536)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > 1048576:
                    raise Rejected("agent_protocol_limit")
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    value = json.loads(line)
                    if value.get("type") == "agent_result":
                        self.agent = value
                        continue
                    if (
                        set(value) != {"type", "id", "method", "params"}
                        or value["type"] != "rpc"
                        or type(value["id"]) is not int
                        or value["id"] in seen
                    ):
                        raise Rejected("agent_protocol_violation")
                    seen.add(value["id"])
                    if len(seen) > self.b["tool_calls"]:
                        raise Rejected("host_rpc_budget")
                    if value["method"] == "run_command" and set(value["params"]) == {
                        "command_id"
                    }:
                        response = self.command(value["params"]["command_id"])
                    elif (
                        value["method"] in ("git_status", "git_diff")
                        and value["params"] == {}
                    ):
                        patch, files = self.diff()
                        response = {
                            "success": True,
                            "output": (
                                json.dumps(files)
                                if value["method"] == "git_status"
                                else patch.decode(errors="replace")
                            )[:32768],
                        }
                    else:
                        raise Rejected("host_rpc_denied")
                    child.stdin.write(
                        canonical({"id": value["id"], "result": response}) + b"\n"
                    )
                    child.stdin.flush()
            child.wait(timeout=2)
            if child.returncode != 0 or not self.agent:
                raise Rejected("agent_interrupted")
        finally:
            selector.close()
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)

    def stop_containers(self, keep_holder=False):
        try:
            live = (
                self.checked(
                    [
                        "ps",
                        "-a",
                        "--filter",
                        "label=vcode.coding.attempt=" + self.job["attempt_id"],
                        "--format",
                        "{{.Names}}",
                    ],
                    cleanup=True,
                )
                .decode()
                .splitlines()
            )
            if any(not n.startswith(self.prefix + "-") for n in live):
                return False
            self.uncertain_creations.difference_update(live)
            targets = [n for n in live if not (keep_holder and n == self.holder)]
            if targets:
                self.checked(["rm", "-f", *targets], cleanup=True)
            after = (
                self.checked(
                    [
                        "ps",
                        "-a",
                        "--filter",
                        "label=vcode.coding.attempt=" + self.job["attempt_id"],
                        "--format",
                        "{{.Names}}",
                    ],
                    cleanup=True,
                )
                .decode()
                .splitlines()
            )
            return (
                not self.uncertain_creations
                and not self.resource_creation_uncertain
                and all(keep_holder and n == self.holder for n in after)
            )
        except Exception:
            return False

    def cleanup(self):
        stopped = self.stop_containers()
        for kind, name in [("network", n) for n in reversed(self.networks)] + (
            [("volume", self.volume)] if self.volume else []
        ):
            try:
                code, _, meta = self.docker([kind, "rm", name], cleanup=True)
                if code and not any(
                    word in meta["stderr"].lower() for word in ("not found", "no such")
                ):
                    stopped = False
            except Exception:
                stopped = False
        return stopped

    def execute(self):
        state, reason, patch, files = "failed", "worker_failure", b"", []
        temporary = tempfile.TemporaryDirectory(prefix="vcode-coding-")
        temp = temporary.name
        artifact_complete = False
        artifact_error = None
        try:
            network = self.setup(temp)
            self.run_agent(network)
            self.remaining()
            final = self.snapshot()
            self.workspace = final
            if not self.agent.get("finished"):
                reason = (
                    "policy_denied" if self.agent.get("denied") else "agent_incomplete"
                )
            else:
                passed = True
                for command in self.profile["repositories"][self.job["repository"]][
                    "commands"
                ]:
                    passed = (
                        self.command(command, validation=True)["success"] and passed
                    )
                after = self.snapshot()
                if after != final:
                    reason = "workspace_changed_during_final_tests"
                elif not passed:
                    reason = "final_tests_failed"
                else:
                    state, reason = "completed", "completed"
            patch, files = self.git.diff(self.snapshot())
        except Stopped:
            state = self.stop_reason or "timed_out"
            reason = "job_" + state
        except Exception as error:
            reason = (
                str(error).split(":")[0]
                if isinstance(error, Rejected)
                else type(error).__name__
            )
        finally:
            self.collecting = True
            self.collection_deadline = time.monotonic() + 15
            mutators_stopped = self.stop_containers(keep_holder=True)
            for record in self.records:
                if record["termination_reason"] == "in_progress":
                    record.update(
                        termination_reason=self.stop_reason or "interrupted",
                        executor_stopped=mutators_stopped,
                        elapsed_ms=round((time.monotonic() - self.started) * 1000),
                    )
            if mutators_stopped and self.git is not None and self.volume is not None:
                try:
                    final_files = self.snapshot()
                    patch, files = self.git.diff(final_files)
                    artifact_complete = True
                    if state == "completed" and final_files != self.workspace:
                        state, reason = "failed", "workspace_changed_after_validation"
                except Exception:
                    artifact_error = "unsafe_missing_or_over_budget_workspace"
                    if state == "completed":
                        state, reason = "failed", "artifact_collection_failed"
            stopped = self.cleanup() and mutators_stopped
            temporary.cleanup()
        if not stopped:
            state, reason = "failed", "cleanup_unconfirmed"
        if not artifact_complete and artifact_error is None:
            artifact_error = "snapshot_not_available"
        patch_hash = self.write("patch.diff", patch)
        command_hash = self.write("commands.json", self.records)
        report = {
            "contract_version": 1,
            "capability": "code.patch",
            **{
                k: self.job[k]
                for k in (
                    "task_id",
                    "attempt_id",
                    "correlation_id",
                    "idempotency_key",
                    "repository",
                    "base_sha",
                )
            },
            "state": state,
            "approved_profile_sha256": sha(canonical(self.profile)),
            "model_id": self.profile["inference"]["model"],
            "command_history_status": "recorded",
            "no_success_claim": state != "completed",
            "termination_reason": reason,
            "execution_stopped": stopped,
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
            "changed_files": files,
            "patch": {
                "path": "patch.diff",
                "sha256": patch_hash,
                "bytes": len(patch),
                "status": "complete" if artifact_complete else "unavailable",
                "error": artifact_error,
            },
            "commands": self.records,
            "tests_not_executed": [
                c
                for c in self.profile["repositories"][self.job["repository"]][
                    "commands"
                ]
                if not any(
                    r["validation"] and r["command_id"] == c for r in self.records
                )
            ],
            "artifact_hashes": {
                "patch.diff": patch_hash,
                "commands.json": command_hash,
            },
            "images": getattr(self, "images", {}),
            "network_evidence": getattr(self, "network_evidence", {}),
            "resource_limits": {
                "budget_scope": "per_unprivileged_container",
                "maximum_simultaneous_containers": 5,
                "maximum_cpus": 4 * self.b["cpus"] + 0.1,
                "maximum_memory_mb": 4 * self.b["memory_mb"] + 64,
                "maximum_pids": 4 * self.b["pids"] + 16,
                "workspace_mb": self.b["workspace_mb"],
            },
            "agent": self.agent,
        }
        digest = self.write("result.json", report)
        self.write(
            "artifacts.json",
            {
                **report["artifact_hashes"],
                "result.json": digest,
                "launch.json": sha((self.artifacts / "launch.json").read_bytes()),
            },
        )
        print(json.dumps(report), flush=True)
        return 0 if state == "completed" else 1


def recover(profile, attempt):
    identifier(attempt)
    root = Path(profile["artifact_directory"]) / attempt
    manifest = read_json(root / "launch.json", 32768)
    if (
        manifest["profile_sha256"] != sha(canonical(profile))
        or manifest["request"]["attempt_id"] != attempt
    ):
        raise Rejected("recovery_binding")
    if (root / "result.json").exists():
        previous = read_json(root / "result.json", 1048576)
        print(json.dumps(previous))
        return 0 if previous["state"] == "completed" else 1
    # Build no scheduler state; reconstruct only exact namespace names recorded by
    # this trusted attempt. Container TTLs also bound execution after launcher SIGKILL.
    obj = object.__new__(Launcher)
    obj.profile = profile
    obj.job = manifest["request"]
    obj.b = obj.job["budgets"]
    obj.started = time.monotonic()
    obj.deadline = obj.started + 30
    obj.stop_reason = None
    obj.collecting = False
    obj.uncertain_creations = set()
    obj.resource_creation_uncertain = False
    obj.capture_started = False
    prefix = "vcode-coding-" + attempt
    names = (
        obj.checked(
            [
                "ps",
                "-a",
                "--filter",
                "label=vcode.coding.attempt=" + attempt,
                "--format",
                "{{.Names}}",
            ],
            cleanup=True,
        )
        .decode()
        .splitlines()
    )
    if any(not n.startswith(prefix + "-") for n in names):
        raise Rejected("foreign_recovery_container")
    obj.names = names
    obj.networks = [prefix + "-private"] + (
        [prefix + "-inference"] if profile["inference_network"] is None else []
    )
    obj.volume = prefix + "-workspace"
    # Missing networks/volumes are valid after partial setup. Verify actual absence.
    stopped = True
    for name in reversed(names):
        try:
            obj.remove(name)
            stopped = obj.absent(name) and stopped
        except Exception:
            stopped = False
    for kind, name in [("network", n) for n in obj.networks] + [("volume", obj.volume)]:
        code, _, meta = obj.docker([kind, "rm", name], cleanup=True)
        if code and not any(
            word in meta["stderr"].lower() for word in ("not found", "no such")
        ):
            stopped = False
    records = (
        read_json(root / "commands.json", 1048576)
        if (root / "commands.json").exists()
        else []
    )
    for record in records:
        if record["termination_reason"] == "in_progress":
            record.update(
                termination_reason="interrupted",
                executor_stopped=stopped,
                exit_code=None,
            )
    obj.artifacts = root
    command_hash = obj.write("commands.json", records)
    report = {
        "contract_version": 1,
        "capability": "code.patch",
        **{
            k: manifest["request"][k]
            for k in (
                "task_id",
                "attempt_id",
                "correlation_id",
                "idempotency_key",
                "repository",
                "base_sha",
            )
        },
        "state": "failed",
        "termination_reason": "interrupted" if stopped else "cleanup_unconfirmed",
        "execution_stopped": stopped,
        "patch": None,
        "changed_files": None,
        "elapsed_ms": None,
        "approved_profile_sha256": manifest["profile_sha256"],
        "model_id": profile["inference"]["model"],
        "command_history_status": "interrupted",
        "commands": records,
        "tests_not_executed": [
            c
            for c in profile["repositories"][manifest["request"]["repository"]][
                "commands"
            ]
            if not any(r["validation"] and r["command_id"] == c for r in records)
        ],
        "artifact_hashes": {"commands.json": command_hash},
        "images": {
            role: {
                "id": profile[role + "_image"],
                "source_sha": None,
                "architecture": None,
            }
            for role in ("agent", "executor")
        },
        "network_evidence": {},
        "resource_limits": {},
        "agent": {},
        "no_success_claim": True,
    }
    obj.artifacts = root
    digest = obj.write("result.json", report)
    obj.write(
        "artifacts.json",
        {
            "result.json": digest,
            "commands.json": command_hash,
            "launch.json": sha((root / "launch.json").read_bytes()),
        },
    )
    print(json.dumps(report))
    return 1


def main():
    global DISPATCH_MAY_HAVE_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "recover"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--request")
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if args.action == "recover":
        DISPATCH_MAY_HAVE_STARTED = True
    profile = load_profile(args.profile, recover=args.action == "recover")
    if args.action == "recover":
        return recover(profile, args.attempt)
    job = request(read_json(args.request, 16384), profile)
    try:
        launcher = Launcher(profile, job)
    except FileExistsError:
        # An existing attempt may still be running. Reject replay without claiming
        # that the original execution is absent or stopped. Only recover can reconcile.
        DISPATCH_MAY_HAVE_STARTED = True
        raise Rejected("attempt_already_exists")
    DISPATCH_MAY_HAVE_STARTED = True
    return launcher.execute()


def entry():
    global DISPATCH_MAY_HAVE_STARTED
    DISPATCH_MAY_HAVE_STARTED = False
    try:
        return main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "state": "failed",
                    "termination_reason": str(error).split(":")[0]
                    if isinstance(error, Rejected)
                    else type(error).__name__,
                    "execution_started": None if DISPATCH_MAY_HAVE_STARTED else False,
                    "dispatch_may_have_started": DISPATCH_MAY_HAVE_STARTED,
                    "execution_stopped": None if DISPATCH_MAY_HAVE_STARTED else True,
                }
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    sys.exit(entry())
