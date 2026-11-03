"""Strict local coding-job v1. The protected operator profile, never model text, grants authority."""

import hashlib
import ipaddress
import json
import os
import re
import stat
import uuid
from pathlib import Path
from urllib.parse import urlsplit

COMMANDS = {
    "python_unittest": [
        "/usr/bin/python3",
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-t",
        ".",
    ],
    "node_test": ["/usr/local/bin/node", "--test"],
    "python_syntax": [
        "/usr/bin/python3",
        "-I",
        "-B",
        "-c",
        "import sys,base64;sys.path.insert(0,'/opt/coding');from snapshot import collect;f=collect();p=[(n,v) for n,v in f.items() if n.endswith('.py')];assert p;[compile(base64.b64decode(v['data']),n,'exec') for n,v in p];print('Python source compilation passed:',len(p))",
    ],
}
TOOLS = [
    "read_file",
    "list_files",
    "search_files",
    "write_file",
    "edit_file",
    "run_command",
    "git_status",
    "git_diff",
]


class Rejected(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise Rejected("unknown_or_missing_fields")


def identifier(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise Rejected("canonical_uuid_required")
    return value


def integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise Rejected("budget_out_of_range")
    return value


def protected(path, directory=False):
    p = Path(path)
    s = p.lstat()
    if (
        not p.is_absolute()
        or stat.S_ISLNK(s.st_mode)
        or s.st_uid != os.getuid()
        or s.st_mode & 0o077
    ):
        raise Rejected("protected_owner_path_required")
    if (directory and not stat.S_ISDIR(s.st_mode)) or (
        not directory and not stat.S_ISREG(s.st_mode)
    ):
        raise Rejected("invalid_path_type")
    return p


def read_json(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Rejected("regular_json_file_required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise Rejected("input_too_large")

    def unique(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise Rejected("duplicate_json_key")
            value[k] = v
        return value

    return json.loads(data, object_pairs_hook=unique)


def load_profile(path, *, recover=False):
    profile = read_json(protected(path), 65536)
    keys(
        profile,
        [
            "contract_version",
            "agent_image",
            "executor_image",
            "docker_context",
            "inference",
            "inference_network",
            "repositories",
            "artifact_directory",
        ],
    )
    if profile["contract_version"] != 1 or type(profile["contract_version"]) is not int:
        raise Rejected("version")
    for k in ("agent_image", "executor_image"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", profile[k]):
            raise Rejected("immutable_image_required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile["docker_context"]):
        raise Rejected("docker_context")
    if profile["inference_network"] is not None and not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,100}", profile["inference_network"]
    ):
        raise Rejected("inference_network")
    protected(profile["artifact_directory"], directory=True)
    inference = profile["inference"]
    keys(inference, ["base_url", "model"])
    u = urlsplit(inference["base_url"])
    ipaddress.IPv4Address(u.hostname)
    if (
        u.scheme != "http"
        or not u.port
        or u.path != "/v1"
        or u.query
        or u.fragment
        or u.username
        or u.password
    ):
        raise Rejected("fixed_ipv4_inference_origin_required")
    if (
        not isinstance(inference["model"], str)
        or not 1 <= len(inference["model"]) <= 200
    ):
        raise Rejected("model")
    if (
        not isinstance(profile["repositories"], dict)
        or not 1 <= len(profile["repositories"]) <= 16
    ):
        raise Rejected("repositories")
    for name, repo in profile["repositories"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise Rejected("repository_id")
        keys(repo, ["path", "base_sha", "commands"])
        if not Path(repo["path"]).is_absolute() or (
            not recover and not Path(repo["path"]).is_dir()
        ):
            raise Rejected("local_repository_required")
        if not re.fullmatch(r"[0-9a-f]{40}", repo["base_sha"]):
            raise Rejected("base_sha")
        if (
            not isinstance(repo["commands"], list)
            or not repo["commands"]
            or len(set(repo["commands"])) != len(repo["commands"])
            or any(c not in COMMANDS for c in repo["commands"])
        ):
            raise Rejected("command_profile")
    return profile


def request(value, profile):
    keys(
        value,
        [
            "contract_version",
            "capability",
            "task_id",
            "attempt_id",
            "correlation_id",
            "idempotency_key",
            "repository",
            "base_sha",
            "instructions",
            "budgets",
        ],
    )
    if (
        type(value["contract_version"]) is not int
        or value["contract_version"] != 1
        or value["capability"] != "code.patch"
    ):
        raise Rejected("unsupported_contract")
    for name in ("task_id", "attempt_id", "correlation_id", "idempotency_key"):
        identifier(value[name])
    repo = profile["repositories"].get(value["repository"])
    if repo is None or value["base_sha"] != repo["base_sha"]:
        raise Rejected("repository_revision_not_enrolled")
    if (
        not isinstance(value["instructions"], str)
        or not value["instructions"].strip()
        or len(value["instructions"].encode()) > 8000
        or "\x00" in value["instructions"]
    ):
        raise Rejected("instructions")
    b = value["budgets"]
    keys(
        b,
        [
            "wall_seconds",
            "command_seconds",
            "cpus",
            "memory_mb",
            "workspace_mb",
            "pids",
            "tool_calls",
        ],
    )
    integer(b["wall_seconds"], 5, 180)
    integer(b["command_seconds"], 1, 30)
    integer(b["memory_mb"], 128, 1024)
    integer(b["workspace_mb"], 8, 64)
    integer(b["pids"], 32, 128)
    integer(b["tool_calls"], 1, 64)
    if type(b["cpus"]) not in (int, float) or not 0.1 <= b["cpus"] <= 2:
        raise Rejected("cpu_budget")
    if b["command_seconds"] > b["wall_seconds"]:
        raise Rejected("command_exceeds_job")
    return value
