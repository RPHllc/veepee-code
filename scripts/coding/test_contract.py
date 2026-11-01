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
                pass

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
