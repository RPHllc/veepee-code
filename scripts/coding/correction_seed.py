"""Read one stopped immutable attempt; no Docker, mutable workspace or caller path."""

import fcntl
import json
import os
import re
import stat
from pathlib import Path

from contract import Rejected, canonical, protected, request, sha

FILES = {
    "launch.json": 32768,
    "result.json": 1048576,
    "commands.json": 1048576,
    "artifacts.json": 65536,
    "creation-state.json": 65536,
    "patch.diff": 4194304,
}
RESULT_FIELDS = {
    "contract_version", "capability", "task_id", "attempt_id", "correlation_id",
    "idempotency_key", "repository", "base_sha", "state", "termination_reason",
    "execution_stopped", "elapsed_ms", "approved_profile_sha256", "model_id",
    "command_history_status", "no_success_claim", "commands", "tests_not_executed",
    "changed_files", "patch", "images", "agent", "network_evidence",
    "resource_limits", "artifact_hashes",
}


def regular_file(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o077 or info.st_size > limit):
            raise Rejected("unsafe_correction_seed_file")
        value = stream.read(limit + 1)
        if len(value) != info.st_size:
            raise Rejected("changed_correction_seed_file")
        return value


def decode(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise Rejected("duplicate_correction_seed_json_key")
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique)


def load(profile, job):
    request(job, profile)
    seed = job["correction_seed"]
    root = protected(Path(profile["artifact_directory"]), directory=True)
    root = protected(root / seed["attempt_id"], directory=True)
    fd = os.open(root / "attempt.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o077):
            raise Rejected("unsafe_correction_seed_lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Rejected("correction_seed_still_running") from None
        files = {name: regular_file(root / name, limit) for name, limit in FILES.items()}
    finally:
        os.close(fd)
    if sha(canonical({k: sha(v) for k, v in files.items()})) != seed["artifact_digest"]:
        raise Rejected("correction_seed_artifact_digest")
    try:
        launch, result, commands, manifest, creation = [
            decode(files[name]) for name in
            ("launch.json", "result.json", "commands.json", "artifacts.json", "creation-state.json")
        ]
        prior = request(launch["request"], profile)
        bindings = ("contract_version", "capability", "task_id", "attempt_id", "correlation_id", "idempotency_key", "repository", "base_sha")
        profile_hash = sha(canonical(profile))
        if (set(launch) != {"request", "creation_journal_version", "profile_sha256", "source_images"}
            or set(result) != RESULT_FIELDS or type(result["contract_version"]) is not int
            or type(launch["creation_journal_version"]) is not int or launch["creation_journal_version"] != 1
            or prior["task_id"] != seed["task_id"] or prior["attempt_id"] != seed["attempt_id"]
            or any(prior[k] != job[k] for k in ("repository", "base_sha"))
            or any(result[k] != prior[k] for k in bindings)
            or launch["profile_sha256"] != profile_hash
            or result["approved_profile_sha256"] != profile_hash
            or launch["source_images"] != [profile["agent_image"], profile["executor_image"]]
            or result["model_id"] != profile["inference"]["model"]
            or result["state"] not in {"failed", "completed", "timed_out"}
            or result["execution_stopped"] is not True
            or result["termination_reason"] in {"cleanup_unconfirmed", "execution_stop_unconfirmed"}
            or result["command_history_status"] != "recorded"
            or result["no_success_claim"] is not (result["state"] != "completed")
            or creation != {"version": 1, "pending": []} or type(creation["version"]) is not int
            or not isinstance(commands, list) or len(commands) > 8 or commands != result["commands"]):
            raise Rejected("correction_seed_identity_or_stop")
        hashes = {name: sha(files[name]) for name in ("commands.json", "patch.diff")}
        if (result["artifact_hashes"] != hashes
            or manifest != {**hashes, **{name: sha(files[name]) for name in ("launch.json", "result.json")}}
            or sha(files["patch.diff"]) != seed["patch_sha256"]
            or result["patch"]["path"] != "patch.diff" or result["patch"]["status"] != "complete"
            or result["patch"].get("error") is not None
            or type(result["patch"]["bytes"]) is not int or result["patch"]["bytes"] != len(files["patch.diff"])
            or result["patch"]["sha256"] != seed["patch_sha256"]):
            raise Rejected("correction_seed_patch_integrity")
        images = result["images"]
        for role in ("agent", "executor"):
            if (images[role]["id"] != profile[role + "_image"]
                or images[role]["architecture"] != "amd64"
                or not isinstance(images[role]["source_sha"], str)
                or not re.fullmatch(r"[a-f0-9]{40}", images[role]["source_sha"])):
                raise Rejected("correction_seed_image_identity")
        if images["agent"]["source_sha"] != images["executor"]["source_sha"]:
            raise Rejected("correction_seed_source_identity")
        return files["patch.diff"], images, result["changed_files"]
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        if isinstance(error, Rejected):
            raise
        raise Rejected("invalid_correction_seed") from None
