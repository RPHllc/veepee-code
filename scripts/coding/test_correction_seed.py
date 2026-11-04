"""Synthetic immutable artifacts and real index-only Git; Docker must not run."""

import base64
import copy
import json
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from contract import Rejected, canonical, load_profile, request, sha
from correction_seed import FILES, load
from git_snapshot import GitSnapshot
from launcher import Launcher, open_attempt_lock


def capture(args, data=None, limit=4194304, env=None, **kwargs):
    result = subprocess.run(args, input=data, capture_output=True, env=env, timeout=10)
    if len(result.stdout) > limit or len(result.stderr) > 4096:
        raise Rejected("test_capture_bound")
    return result.returncode, result.stdout, {"stderr": result.stderr.decode(), "truncated": False}


class CorrectionSeed(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="coding-seed-unit-")
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        for path, data in {"app.py": "def add(a,b):\n    return a - b\n", "README.md": "trusted requirements\n", "tests/test_app.py": "trusted test\n"}.items():
            target = self.repo / path;target.parent.mkdir(parents=True, exist_ok=True);target.write_text(data)
        self.git("add", ".")
        self.git("-c", "user.name=Synthetic", "-c", "user.email=test@example.invalid", "commit", "-qm", "original base")
        self.base = self.git("rev-parse", "HEAD").decode().strip()
        self.artifacts = self.root / "artifacts";self.artifacts.mkdir(mode=0o700)
        self.profile = {
            "contract_version": 1, "agent_image": "sha256:" + "a" * 64,
            "executor_image": "sha256:" + "b" * 64, "docker_context": "default",
            "inference": {"base_url": "http://192.0.2.1:8000/v1", "model": "synthetic"},
            "inference_network": None, "artifact_directory": str(self.artifacts),
            "repositories": {"fixture": {"path": str(self.repo), "base_sha": self.base,
                "commands": ["python_unittest"], "allowed_paths": ["app.py", "helper.py"], "allow_correction_seeds": True}},
        }
        self.prior = self.job()
        self.current = self.job()
        self.directory = self.artifacts / self.prior["attempt_id"];self.directory.mkdir(mode=0o700)
        (self.directory / "attempt.lock").write_bytes(b"");(self.directory / "attempt.lock").chmod(0o600)
        self.make_seed()

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.repo), *args])

    def job(self):
        return {"contract_version": 1, "capability": "code.patch", "repository": "fixture", "base_sha": self.base,
            **{key: str(uuid.uuid4()) for key in ("task_id", "attempt_id", "correlation_id", "idempotency_key")},
            "instructions": "Correct the existing implementation and run genuine tests.",
            "budgets": {"wall_seconds": 60, "command_seconds": 10, "cpus": 1, "memory_mb": 256, "workspace_mb": 8, "pids": 64, "tool_calls": 32}}

    def make_seed(self, changes=None):
        changes = changes or {"app.py": "# seeded implementation\ndef add(a,b):\n    return a + b\n"}
        for path, data in changes.items():(self.repo / path).write_text(data)
        patch = self.git("diff", "--binary", "--full-index")
        self.git("add", ".")
        tree = self.git("write-tree").decode().strip()
        self.git("reset", "--hard", "-q", "HEAD")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = GitSnapshot(capture, str(self.repo), self.base, Path(tmp) / "git")
            # Test construction intentionally permits the negative protected-file fixture.
            files = snapshot.apply_seed(patch, list(snapshot.base) + ["helper.py"], tree)
            inventory = snapshot.diff(files)[1]
        images = {role: {"id": self.profile[role + "_image"], "architecture": "amd64", "source_sha": "c" * 40} for role in ("agent", "executor")}
        launch = {"request": self.prior, "profile_sha256": sha(canonical(self.profile)), "creation_journal_version": 1, "source_images": [self.profile["agent_image"], self.profile["executor_image"]]}
        self.report = {**self.prior, "state": "failed", "termination_reason": "final_tests_failed", "execution_stopped": True,
            "elapsed_ms": 1, "approved_profile_sha256": sha(canonical(self.profile)), "model_id": "synthetic",
            "command_history_status": "recorded", "no_success_claim": True, "commands": [], "tests_not_executed": [],
            "changed_files": inventory, "patch": {"path": "patch.diff", "sha256": sha(patch), "bytes": len(patch), "status": "complete", "error": None},
            "images": images, "agent": {}, "network_evidence": {}, "resource_limits": {}, "artifact_hashes": {}}
        for key in ("instructions", "budgets"):self.report.pop(key)
        self.files = {"launch.json": canonical(launch), "commands.json": canonical([]), "patch.diff": patch,
            "creation-state.json": canonical({"version": 1, "pending": []})}
        self.reseal(tree)

    def reseal(self, tree=None):
        self.report["artifact_hashes"] = {name: sha(self.files[name]) for name in ("commands.json", "patch.diff")}
        self.files["result.json"] = canonical(self.report)
        self.files["artifacts.json"] = canonical({name: sha(self.files[name]) for name in ("commands.json", "patch.diff", "launch.json", "result.json")})
        for name, data in self.files.items():(self.directory / name).write_bytes(data);(self.directory / name).chmod(0o600)
        self.current["correction_seed"] = {"task_id": self.prior["task_id"], "attempt_id": self.prior["attempt_id"],
            "artifact_digest": sha(canonical({name: sha(value) for name, value in self.files.items()})),
            "patch_sha256": sha(self.files["patch.diff"]), "tree_sha": tree or self.current["correction_seed"]["tree_sha"]}

    def assert_no_docker(self):
        value = request(self.current, self.profile)
        launcher = Launcher(self.profile, value)
        launcher.capture = capture
        launcher.docker = lambda *a, **kw: self.fail("Docker called for rejected seed")
        try:
            with self.assertRaises((Rejected, OSError)):
                launcher.execute()
        finally:launcher.lock.close()

    def test_valid_seed_uses_original_base_and_full_diff(self):
        patch, images, inventory = load(self.profile, request(self.current, self.profile))
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = GitSnapshot(capture, str(self.repo), self.base, Path(tmp) / "git")
            original = copy.deepcopy(snapshot.base)
            files = snapshot.apply_seed(patch, ["app.py"], self.current["correction_seed"]["tree_sha"])
            self.assertEqual(snapshot.base, original)
            self.assertEqual(snapshot.diff(files)[1], inventory)
            self.assertIn(b"seeded implementation", base64.b64decode(files["app.py"]["data"]))
            files["app.py"]["data"] = base64.b64encode(b"# seeded implementation\ndef add(a,b):\n    return a + b + 0\n").decode()
            final, _ = snapshot.diff(files)
            self.assertIn(b"-    return a - b", final)
            self.assertIn(b"+    return a + b + 0", final)
            self.assertEqual(snapshot.git(snapshot.root, ["rev-parse", "HEAD"]).decode().strip(), self.base)

    def test_wrong_digest_refuses_before_docker(self):
        self.current["correction_seed"]["artifact_digest"] = "0" * 64;self.assert_no_docker()

    def test_seed_reference_is_closed_and_canonical(self):
        for key, value in (("path", "/arbitrary"), ("tree_sha", "a" * 64), ("patch_sha256", "A" * 64), ("attempt_id", self.current["attempt_id"])):
            wire = copy.deepcopy(self.current);wire["correction_seed"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):request(wire, self.profile)

    def test_cross_repository_refuses_before_docker(self):
        launch = json.loads(self.files["launch.json"]);launch["request"]["repository"] = "other"
        self.files["launch.json"] = canonical(launch);self.report["repository"] = "other";self.reseal();self.assert_no_docker()

    def test_running_state_refuses_before_docker(self):
        self.report["state"] = "running";self.reseal();self.assert_no_docker()

    def test_safely_stopped_timed_out_complete_seed_is_accepted(self):
        self.report.update(state="timed_out", termination_reason="job_timed_out")
        self.reseal()
        patch, images, inventory = load(self.profile, request(self.current, self.profile))
        self.assertEqual(sha(patch), self.current["correction_seed"]["patch_sha256"])
        self.assertEqual(inventory, self.report["changed_files"])

    def test_uncertain_timed_out_seed_refuses_before_docker(self):
        self.report.update(state="timed_out", termination_reason="job_timed_out", execution_stopped=False)
        self.reseal();self.assert_no_docker()

    def test_wrong_tree_refuses_before_docker(self):
        self.current["correction_seed"]["tree_sha"] = "0" * 40;self.assert_no_docker()

    def test_wrong_patch_hash_refuses_before_docker(self):
        self.current["correction_seed"]["patch_sha256"] = "0" * 64;self.assert_no_docker()

    def test_cross_base_refuses_before_docker(self):
        launch = json.loads(self.files["launch.json"]);launch["request"]["base_sha"] = "0" * 40
        self.files["launch.json"] = canonical(launch);self.report["base_sha"] = "0" * 40;self.reseal();self.assert_no_docker()

    def test_protected_delta_refuses_before_docker(self):
        self.make_seed({"README.md": "forged requirements\n"});self.assert_no_docker()

    def test_missing_file_refuses_before_docker(self):
        (self.directory / "patch.diff").unlink();self.assert_no_docker()

    def test_incomplete_patch_refuses_before_docker(self):
        self.report["patch"]["status"] = "unavailable";self.reseal();self.assert_no_docker()

    def test_uncertain_stop_refuses_before_docker(self):
        self.report["execution_stopped"] = False;self.reseal();self.assert_no_docker()

    def test_pending_creation_refuses_before_docker(self):
        self.files["creation-state.json"] = canonical({"version": 1, "pending": ["unknown"]});self.reseal();self.assert_no_docker()

    def test_active_prior_lock_refuses_before_docker(self):
        with open_attempt_lock(self.directory):self.assert_no_docker()

    def test_symlink_refuses_before_docker(self):
        (self.directory / "patch.diff").unlink();(self.directory / "patch.diff").symlink_to(self.repo / "app.py");self.assert_no_docker()

    def test_profile_flag_requires_literal_paths_and_preserves_omitted_bytes(self):
        legacy = copy.deepcopy(self.profile);repo = legacy["repositories"]["fixture"]
        repo.pop("allowed_paths");repo.pop("allow_correction_seeds")
        path = self.root / "profile.json";path.write_bytes(canonical(legacy));path.chmod(0o600)
        self.assertEqual(canonical(load_profile(path)), canonical(legacy))
        repo["allow_correction_seeds"] = True;path.write_bytes(canonical(legacy))
        with self.assertRaises(Rejected):load_profile(path)
        with self.assertRaises(Rejected):request(self.current, {**self.profile, "repositories": {"fixture": {**self.profile["repositories"]["fixture"], "allow_correction_seeds": False}}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
