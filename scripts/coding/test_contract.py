"""Fast host contract/authorization tests; no Docker or model access."""

import copy, json, tempfile, unittest, uuid
from pathlib import Path
from contract import Rejected, load_profile, request, read_json


class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir(mode=0o700)
        self.repo = root / "repo"
        self.repo.mkdir()
        self.value = {
            "contract_version": 1,
            "capability": "code.patch",
            "repository": "fixture",
            "base_sha": "a" * 40,
            "instructions": "Fix the test.",
            **{
                k: str(uuid.uuid4())
                for k in ("task_id", "attempt_id", "correlation_id", "idempotency_key")
            },
            "budgets": {
                "wall_seconds": 60,
                "command_seconds": 10,
                "cpus": 1,
                "memory_mb": 256,
                "workspace_mb": 16,
                "pids": 64,
                "tool_calls": 32,
            },
        }
        self.profile = {
            "contract_version": 1,
            "agent_image": "sha256:" + "a" * 64,
            "executor_image": "sha256:" + "b" * 64,
            "docker_context": "default",
            "inference": {"base_url": "http://10.0.0.2:8000/v1", "model": "fixed"},
            "inference_network": None,
            "artifact_directory": str(self.artifacts),
            "repositories": {
                "fixture": {
                    "path": str(self.repo),
                    "base_sha": "a" * 40,
                    "commands": ["python_unittest"],
                }
            },
        }
        self.path = root / "profile.json"

    def tearDown(self):
        self.temp.cleanup()

    def profile_file(self):
        self.path.write_text(json.dumps(self.profile))
        self.path.chmod(0o600)
        return self.path

    def test_valid_contract_has_no_client_authority_fields(self):
        p = load_profile(self.profile_file())
        self.assertEqual(request(self.value, p), self.value)
        for field in (
            "allowed_tools",
            "repository_path",
            "image",
            "endpoint",
            "command",
            "credentials",
            "shell",
            "permissions",
        ):
            with self.subTest(field=field):
                value = {**self.value, field: "attacker"}
                with self.assertRaises(Rejected):
                    request(value, p)

    def test_repository_and_revision_require_explicit_enrollment(self):
        for field, value in [
            ("repository", "unknown"),
            ("base_sha", "b" * 40),
            ("contract_version", True),
            ("contract_version", 2),
            ("capability", "github.push"),
        ]:
            with self.subTest(field=field):
                with self.assertRaises(Rejected):
                    request({**self.value, field: value}, self.profile)

    def test_output_budget_is_protected_bounded_and_backward_compatible(self):
        self.assertNotIn("max_output_tokens", load_profile(self.profile_file())["inference"])
        self.profile["inference"]["max_output_tokens"] = 8192
        p = load_profile(self.profile_file())
        self.assertEqual(p["inference"]["max_output_tokens"], 8192)
        with self.assertRaises(Rejected):
            request({**self.value, "max_output_tokens": 8192}, p)
        for value in (True, 0, 255, 8193, "8192", None):
            with self.subTest(value=value):
                self.profile["inference"]["max_output_tokens"] = value
                with self.assertRaises(Rejected):
                    load_profile(self.profile_file())

    def test_budgets_cannot_expand_ceiling_or_hide_types(self):
        for field, value in [
            ("wall_seconds", 181),
            ("wall_seconds", True),
            ("command_seconds", 31),
            ("cpus", 3),
            ("memory_mb", 4096),
            ("workspace_mb", 1000),
            ("pids", 1024),
            ("tool_calls", 65),
        ]:
            with self.subTest(field=field):
                bad = copy.deepcopy(self.value)
                bad["budgets"][field] = value
                with self.assertRaises(Rejected):
                    request(bad, self.profile)

    def test_protected_operator_profile_and_immutable_images(self):
        path = self.profile_file()
        path.chmod(0o644)
        with self.assertRaises(Rejected):
            load_profile(path)
        path.chmod(0o600)
        self.profile["agent_image"] = "vcode:latest"
        self.profile_file()
        with self.assertRaises(Rejected):
            load_profile(path)

    def test_arbitrary_shell_commands_and_network_origins_are_not_profiles(self):
        self.profile["repositories"]["fixture"]["commands"] = ["bash"]
        self.profile_file()
        with self.assertRaises(Rejected):
            load_profile(self.path)

    def test_larger_wall_budget_requires_protected_repository_enrollment(self):
        value = copy.deepcopy(self.value)
        value["budgets"]["wall_seconds"] = 181
        with self.assertRaises(Rejected):
            request(value, load_profile(self.profile_file()))
        self.profile["repositories"]["fixture"]["max_wall_seconds"] = 600
        profile = load_profile(self.profile_file())
        value["budgets"]["wall_seconds"] = 600
        self.assertEqual(request(value, profile), value)
        value["budgets"]["wall_seconds"] = 601
        with self.assertRaises(Rejected):
            request(value, profile)
        for bad in (True, 0, 601, "600"):
            self.profile["repositories"]["fixture"]["max_wall_seconds"] = bad
            with self.assertRaises(Rejected):
                load_profile(self.profile_file())
        self.profile["repositories"]["fixture"]["commands"] = ["python_unittest"]
        for url in (
            "http://evil.invalid:8000/v1",
            "http://10.0.0.2:8000/arbitrary",
            "http://user:pass@10.0.0.2:8000/v1",
        ):
            self.profile["inference"]["base_url"] = url
            self.profile_file()
            with self.assertRaises((Rejected, ValueError)):
                load_profile(self.path)

    def test_literal_write_paths_require_protected_profile_and_cannot_be_requested(self):
        self.assertNotIn("allowed_paths", load_profile(self.profile_file())["repositories"]["fixture"])
        self.profile["repositories"]["fixture"]["allowed_paths"] = ["app.py", "src/new.py"]
        self.assertEqual(load_profile(self.profile_file())["repositories"]["fixture"]["allowed_paths"], ["app.py", "src/new.py"])
        for field in ("allowed_paths", "writable_paths"):
            with self.assertRaises(Rejected):
                request({**self.value, field: ["tests/test_app.py"]}, self.profile)
        for paths in ([], ["app.py", "app.py"], ["*.py"], ["/app.py"], ["src/../app.py"], ["src//app.py"], ["tests/"], [".git/config"], [".env"], ["app.py.coding-next"], ["app.py\n"], [None], "app.py", ["x" * 241], [f"file{i}.py" for i in range(33)]):
            with self.subTest(paths=paths):
                self.profile["repositories"]["fixture"]["allowed_paths"] = paths
                with self.assertRaises(Rejected):
                    load_profile(self.profile_file())

    def test_duplicate_json_keys_and_input_bounds(self):
        self.path.write_text('{"contract_version":1,"contract_version":2}')
        with self.assertRaises(Rejected):
            read_json(self.path, 100)
        self.path.write_bytes(b"x" * 101)
        with self.assertRaises(Rejected):
            read_json(self.path, 100)


class ControlPlaneFailures(unittest.TestCase):
    def test_lost_create_ack_is_not_a_confirmed_stop(self):
        from launcher import Launcher

        o = object.__new__(Launcher)
        o.job = {"attempt_id": str(uuid.uuid4())}
        o.prefix = "vcode-coding-" + o.job["attempt_id"]
        o.holder = None
        name = o.prefix + "-agent"
        o.uncertain_creations = {name}
        o.resource_creation_uncertain = False
        o.creation_quiescent = lambda: True
        o.checked = lambda args, **kwargs: b""
        self.assertFalse(o.stop_containers())
        calls = []

        def known(args, **kwargs):
            calls.append(args)
            return name.encode() if len(calls) == 1 else b""

        o.checked = known
        self.assertTrue(o.stop_containers())
        self.assertEqual(calls[1], ["rm", "-f", name])

    def test_post_dispatch_error_cannot_claim_no_execution(self):
        import launcher, contextlib, io
        from unittest.mock import patch

        output = io.StringIO()
        argv = ["launcher.py", "run", "--profile", "unused", "--request", "unused"]

        class FakeLauncher:
            def __init__(self, *args):
                from unittest.mock import Mock

                self.lock = Mock()

            def execute(self):
                raise OSError("synthetic artifact failure after dispatch")

        with (
            patch.object(launcher, "load_profile", return_value={}),
            patch.object(launcher, "read_json", return_value={}),
            patch.object(launcher, "request", return_value={}),
            patch.object(launcher, "Launcher", FakeLauncher),
            patch("sys.argv", argv),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(launcher.entry(), 2)
        value = json.loads(output.getvalue())
        self.assertTrue(value["dispatch_may_have_started"])
        self.assertIsNone(value["execution_started"])
        self.assertIsNone(value["execution_stopped"])


class RecoveryPersistence(unittest.TestCase):
    def setUp(self):
        from launcher import Launcher

        self.fixture = Contract()
        self.fixture.setUp()
        self.profile, self.job = self.fixture.profile, self.fixture.value
        self.obj = Launcher(self.profile, self.job)
        self.root = self.obj.artifacts
        self.obj.lock.close()

    def tearDown(self):
        self.fixture.tearDown()

    def recovery(self):
        import contextlib, io
        from unittest.mock import patch
        from launcher import Launcher, recover

        def docker(obj, args, **kwargs):
            return 0, b"", {"stderr": "", "truncated": False}

        output = io.StringIO()
        with (
            patch.object(Launcher, "docker", docker),
            contextlib.redirect_stdout(output),
        ):
            code = recover(self.profile, self.job["attempt_id"])
        return code, json.loads(output.getvalue())

    def interrupt_write(self, name, value):
        """SIGKILL after staging fsync, immediately before atomic publication."""
        import subprocess, sys, time

        marker = self.root / "staged.marker"
        marker.unlink(missing_ok=True)
        code = """
import json,os,sys,time
from pathlib import Path
from launcher import Launcher
obj=object.__new__(Launcher);obj.artifacts=Path(sys.argv[1])
def pause(*args):
 (obj.artifacts/'staged.marker').write_text('ready')
 while True:time.sleep(.05)
os.replace=pause
obj.write(sys.argv[2],json.loads(sys.argv[3]))
"""
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(self.root), name, json.dumps(value)],
            cwd=Path(__file__).parent,
        )
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists())
        finally:
            child.kill()
            child.wait(timeout=5)
        self.assertTrue(list(self.root.glob("." + name + ".*.next")))

    def test_interrupted_command_persistence_uses_only_published_ledger(self):
        old = [
            {
                "command_id": "python_unittest",
                "validation": False,
                "termination_reason": "in_progress",
                "exit_code": None,
            }
        ]
        self.obj.write("commands.json", old)
        self.interrupt_write(
            "commands.json",
            [{**old[0], "termination_reason": "exited", "exit_code": 0}],
        )
        (self.root / "commands.json.next").write_bytes(b"legacy partial write")
        _, report = self.recovery()
        self.assertEqual(report["commands"][0]["termination_reason"], "interrupted")
        self.assertIsNone(report["commands"][0]["exit_code"])
        self.assertTrue(report["execution_stopped"])
        before = {p.name: p.read_bytes() for p in self.root.glob("*.json")}
        self.assertEqual(self.recovery()[1], report)
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.root.glob("*.json")}
        )
        self.assertEqual(
            (self.root / "commands.json.next").read_bytes(), b"legacy partial write"
        )

    def test_interrupted_result_persistence_never_promotes_staged_success(self):
        self.interrupt_write(
            "result.json", {"state": "completed", "execution_stopped": True}
        )
        (self.root / "result.json.next").write_text('{"state":"completed"}')
        _, report = self.recovery()
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["termination_reason"], "interrupted")
        self.assertTrue(report["no_success_claim"])
        self.assertEqual(self.recovery()[1], report)

    def test_interrupted_manifest_persistence_preserves_published_result(self):
        from contract import sha

        # First make a valid published recovery result/ledger, then interrupt its
        # manifest persistence. Recovery must finalize hashes without rewriting them.
        _, report = self.recovery()
        # Trusted published-completion fixture: this checks byte preservation,
        # not real command execution (the Linux suite supplies process evidence).
        report.update(
            state="completed", termination_reason="completed", no_success_claim=False
        )
        self.obj.write("result.json", report)
        result = (self.root / "result.json").read_bytes()
        ledger = (self.root / "commands.json").read_bytes()
        manifest = read_json(self.root / "artifacts.json", 65536)
        manifest["result.json"] = sha(result)
        (self.root / "artifacts.json").unlink()
        self.interrupt_write("artifacts.json", manifest)
        (self.root / "artifacts.json.next").write_bytes(b"partial")
        code, repeated = self.recovery()
        self.assertEqual(code, 0)
        self.assertEqual(repeated, report)
        self.assertEqual((self.root / "result.json").read_bytes(), result)
        self.assertEqual((self.root / "commands.json").read_bytes(), ledger)
        self.assertEqual(read_json(self.root / "artifacts.json", 65536), manifest)
        self.assertEqual(manifest["result.json"], sha(result))
        # Corrupted published evidence must be rejected, never repaired by overwrite.
        (self.root / "commands.json").write_bytes(b"[] ")
        with self.assertRaisesRegex(Rejected, "published_artifact_integrity"):
            self.recovery()
        self.assertEqual((self.root / "commands.json").read_bytes(), b"[] ")

    def test_recovery_refuses_live_attempt_lock(self):
        from launcher import open_attempt_lock

        with open_attempt_lock(self.root):
            with self.assertRaisesRegex(Rejected, "attempt_in_progress"):
                self.recovery()
        self.assertFalse((self.root / "result.json").exists())

    def test_missing_or_pending_creation_journal_cannot_confirm_stop(self):
        for state in (
            None,
            {"version": True, "pending": []},
            {"version": 1, "pending": [{"id": "unacknowledged"}]},
        ):
            with self.subTest(state=state):
                if state is None:
                    (self.root / "creation-state.json").unlink(missing_ok=True)
                else:
                    self.obj.write("creation-state.json", state)
                _, report = self.recovery()
                self.assertEqual(report["termination_reason"], "cleanup_unconfirmed")
                self.assertFalse(report["execution_stopped"])
                for name in ("result.json", "artifacts.json", "commands.json"):
                    (self.root / name).unlink(missing_ok=True)


class BoundedDiagnosticCapture(unittest.TestCase):
    def capture(self, script, **kwargs):
        import subprocess, sys, time
        from unittest.mock import patch
        from launcher import Launcher
        obj = object.__new__(Launcher)
        obj.collecting = False
        obj.stop_reason = None
        obj.deadline = time.monotonic() + 5
        children = []
        popen = subprocess.Popen

        def start(*args, **options):
            child = popen(*args, **options)
            children.append(child)
            return child

        try:
            with patch.object(subprocess, "Popen", side_effect=start):
                return obj.capture([sys.executable, "-c", script], timeout=3, **kwargs)
        finally:
            for child in children:
                for stream in (child.stdin, child.stdout, child.stderr):
                    if stream:
                        stream.close()

    def test_large_stderr_preserves_prefix_and_final_root_cause_in_same_byte_cap(self):
        import hashlib
        stdout = bytes(range(256)) * 20
        prefix = b"initial context\n" + b"p" * (2048 - 16)
        final = b"\nFINAL ROOT CAUSE ERROR\n"
        suffix = b"t" * (2048 - len(final)) + final
        self.assertEqual(len(prefix), 2048)
        self.assertEqual(len(suffix), 2048)
        code, output, meta = self.capture(
            "import sys;sys.stdout.buffer.write(bytes(range(256))*20);"
            + "sys.stderr.buffer.write(" + repr(prefix) + "+b'middle\\n'*20000+" + repr(suffix) + ");sys.exit(1)"
        )
        self.assertEqual(code, 1)
        self.assertEqual(output, stdout)
        self.assertEqual(hashlib.sha256(output).digest(), hashlib.sha256(stdout).digest())
        self.assertEqual(meta["stderr"].encode(), prefix + suffix)
        self.assertEqual(len(meta["stderr"].encode()), 4096)
        self.assertTrue(meta["truncated"])

    def test_short_stderr_and_stdout_truncation_semantics_are_unchanged(self):
        code, output, meta = self.capture(
            "import sys;sys.stdout.buffer.write(b'0123456789');sys.stderr.write('short error')",
            limit=4, truncate=True,
        )
        self.assertEqual((code, output, meta), (0, b"0123", {"stderr": "short error", "truncated": True}))
        _, output, meta = self.capture("import sys;sys.stdout.write('ok');sys.stderr.write('short error')")
        self.assertEqual((output, meta), (b"ok", {"stderr": "short error", "truncated": False}))

    def test_oversized_binary_stdout_still_rejects_without_truncation_permission(self):
        with self.assertRaisesRegex(Rejected, "bounded_output_exceeded"):
            self.capture("import sys;sys.stdout.buffer.write(b'x'*10000)", limit=64)


class ExtendedRuntimeDeadlines(unittest.TestCase):
    def test_egress_guard_uses_bounded_job_lifetime(self):
        import os, runpy, subprocess, sys
        from unittest.mock import patch
        script = Path(__file__).with_name("netguard.py")
        for seconds in (180, 600):
            with patch.object(sys, "argv", [str(script), "192.0.2.1", "8000", "172.30.0.0/16", str(seconds)]), patch.object(subprocess, "run") as rules, patch.object(os, "execvp") as execute:
                runpy.run_path(str(script))
                self.assertTrue(rules.called)
                execute.assert_called_once_with("timeout", ["timeout", "--signal=KILL", str(seconds), "sleep", str(seconds + 1)])
        with patch.object(sys, "argv", [str(script), "192.0.2.1", "8000", "172.30.0.0/16", "601"]), patch.object(subprocess, "run") as rules, patch.object(os, "execvp") as execute:
            with self.assertRaises(AssertionError):
                runpy.run_path(str(script))
            rules.assert_not_called(); execute.assert_not_called()

    def test_nonzero_agent_exit_keeps_failure_evidence_and_cannot_claim_completion(self):
        from launcher import Launcher
        obj = object.__new__(Launcher)
        for finished in (False, True):
            obj.agent = {"finished": finished, "denied": False, "error": "internal_failure", "events": []}
            obj.accept_agent_exit(1)
            self.assertFalse(obj.agent["finished"])
            self.assertEqual(obj.agent["error"], "internal_failure")
        obj.agent = {"finished": True, "denied": True}
        obj.accept_agent_exit(1)
        self.assertTrue(obj.agent["denied"])
        self.assertFalse(obj.agent["finished"])
        obj.agent = {}
        with self.assertRaisesRegex(Rejected, "agent_interrupted"):
            obj.accept_agent_exit(1)

    def test_agent_budget_reserves_validation_and_sealing_without_increasing_deadline(self):
        from launcher import Launcher
        obj = object.__new__(Launcher)
        obj.job = {"repository": "fixture"}
        obj.profile = {"repositories": {"fixture": {"commands": ["python_unittest", "python_syntax"]}}}
        obj.b = {"command_seconds": 30}
        for remaining in (180, 600):
            obj.remaining = lambda: remaining
            seconds = obj.agent_seconds()
            self.assertGreaterEqual(seconds, 1)
            self.assertLess(seconds, remaining)
            self.assertEqual(seconds + 75, remaining)
        for remaining in (5, 60, 75):
            obj.remaining = lambda: remaining
            with self.assertRaisesRegex(Rejected, "insufficient_finalization_budget"):
                obj.agent_seconds()
        obj.remaining = lambda: 600
        self.assertEqual(obj.agent_seconds(), 525)

    def test_workspace_lease_covers_same_deadline_and_artifact_grace(self):
        from launcher import Launcher
        obj = object.__new__(Launcher)
        obj.prefix = "test-fixed-identity"
        obj.profile = {"executor_image": "sha256:" + "a" * 64}
        obj.flags = lambda name: ["--name", name]
        obj.mount = lambda: []
        for remaining in (60, 180, 600):
            obj.remaining = lambda: remaining
            command = obj.workspace_lease_command()
            self.assertEqual(command[-5:], ["timeout", "--signal=KILL", str(remaining + 20), "sleep", str(remaining + 21)])
            self.assertEqual(obj.holder, obj.prefix + "-workspace-lease")


if __name__ == "__main__":
    unittest.main(verbosity=2)
