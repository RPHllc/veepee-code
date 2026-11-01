#!/usr/bin/env python3
"""Opt-in bounded two-task real-model evaluation; only disposable synthetic repositories.

Does not load secrets, change shared models or use production repository mounts.
Records actual results and checks a separately applied patch with external oracles.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-image", required=True)
    parser.add_argument("--executor-image", required=True)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    root = Path(args.output_directory).resolve()
    root.mkdir(mode=0o700)
    os.environ["CODING_DOCKER_CONTEXT"] = args.docker_context
    os.environ["CODING_EVIDENCE_DIR"] = str(root / "evidence")
    from test_linux import CodingLinux

    CodingLinux.agent, CodingLinux.executor = args.agent_image, args.executor_image
    CodingLinux.endpoint, CodingLinux.network = args.inference_url, None
    cases = [
        (
            "single",
            "Fix the add function in app.py so it correctly adds two numbers. Read the code and test, change only the implementation, run python_unittest and python_syntax, and inspect the diff. Do not change tests or install packages.",
            None,
        ),
        (
            "multi",
            "Fix add(a,b) in app.py to add and double(x) in helper.py to return twice x. Read the code/tests, change both implementations, preserve all tests, run python_unittest and python_syntax and inspect the diff. Do not install dependencies.",
            {
                "app.py": "def add(a,b):\n    return a - b\n",
                "helper.py": "def double(x):\n    return x\n",
                "tests/__init__.py": "",
                "tests/test_app.py": "import unittest\nfrom app import add\nfrom helper import double\nclass Test(unittest.TestCase):\n def test_both(self): self.assertEqual(add(2,3),5); self.assertEqual(double(4),8)\n",
            },
        ),
    ]
    summary = []
    for label, instructions, files in cases:
        fixture = CodingLinux("test_01_repair_and_actual_exit_codes")
        fixture.setUp()
        try:
            fixture.setup_job(
                instructions, files, {"wall_seconds": 120, "command_seconds": 15}
            )
            profile = json.loads(fixture.profile.read_text())
            profile["inference"]["model"] = args.model
            profile["repositories"]["synthetic"]["commands"].append("python_syntax")
            fixture.profile.write_text(json.dumps(profile))
            started = time.monotonic()
            report = fixture.run_job()
            functional = False
            oracle_error = None
            if report.get("state") == "completed":
                try:
                    fixture.assert_patch(report)
                    if label == "multi":
                        subprocess.run(
                            [
                                __import__("sys").executable,
                                "-B",
                                "-c",
                                "from helper import double; assert double(12)==24",
                            ],
                            cwd=fixture.repo,
                            check=True,
                        )
                    functional = True
                except Exception as error:
                    oracle_error = type(error).__name__
            item = {
                "case": label,
                "state": report.get("state"),
                "termination_reason": report.get("termination_reason"),
                "functional_correctness": functional,
                "observed_seconds": round(time.monotonic() - started, 3),
                "worker_elapsed_ms": report.get("elapsed_ms"),
                "model_diagnostics": report.get("agent", {}).get("diagnostics"),
                "base_sha": fixture.base,
                "attempt_id": fixture.job["attempt_id"],
                "images": report.get("images"),
                "patch": report.get("patch"),
                "oracle_error": oracle_error,
            }
            (root / (label + "-report.json")).write_text(json.dumps(report, indent=2))
            summary.append(item)
        finally:
            fixture.tearDown()
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if all(v["functional_correctness"] for v in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
