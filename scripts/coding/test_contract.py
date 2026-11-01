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


if __name__ == "__main__":
    unittest.main(verbosity=2)
