"""Trusted Git object import/diff: never checkout filters, hooks, repository config or shell."""

import base64
import os
import re
import shutil
from pathlib import Path
from contract import Rejected, canonical, sha

ENV = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": "/nonexistent",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_ALLOW_PROTOCOL": "",
    "LC_ALL": "C",
}


def valid(path):
    return (
        bool(re.fullmatch(r"[a-zA-Z0-9_./ -]{1,240}", path))
        and not path.startswith("/")
        and all(
            p not in ("", ".", "..")
            and not p.startswith("-")
            and p.lower() != ".git"
            and not re.fullmatch(
                r"(?:\.env(?:\..*)?|id_rsa|id_ed25519|credentials|secrets(?:\..*)?)",
                p,
                re.I,
            )
            for p in path.split("/")
        )
    )


class GitSnapshot:
    def __init__(self, run, source, revision, target):
        self.run = run
        self.root = Path(target)
        self.gitbin = shutil.which("git")
        if not self.gitbin:
            raise Rejected("git_required")
        self.root.mkdir()
        self.git(self.root, ["-c", "init.templateDir=", "init", "-q"])
        if self.git(source, ["cat-file", "-t", revision]).strip() != b"commit":
            raise Rejected("exact_commit_required")
        entries = self.git(
            source,
            ["ls-tree", "-r", "-t", "-l", "-z", "--full-tree", revision],
            limit=1048576,
        )
        self.base = {}
        total = 0
        root_tree = (
            self.git(source, ["rev-parse", revision + "^{tree}"]).decode().strip()
        )
        objects = {root_tree: "tree", revision: "commit"}
        for item in entries.split(b"\0"):
            if not item:
                continue
            info, rawpath = item.split(b"\t", 1)
            mode, kind, oid, size = info.decode().split()
            path = rawpath.decode("ascii")
            if not valid(path):
                raise Rejected("unsafe_repository_path")
            if kind == "tree":
                objects[oid] = "tree"
                continue
            if kind != "blob" or mode not in ("100644", "100755") or int(size) > 262144:
                raise Rejected("links_submodules_special_or_large_files_denied")
            total += int(size)
            if total > 2097152 or len(self.base) >= 512:
                raise Rejected("repository_snapshot_budget")
            data = self.git(source, ["cat-file", "blob", oid], limit=262144)
            self.base[path] = {"mode": mode, "data": base64.b64encode(data).decode()}
            objects[oid] = "blob"
        for oid, kind in objects.items():
            raw = self.git(source, ["cat-file", kind, oid], limit=2097152)
            actual = (
                self.git(
                    self.root, ["hash-object", "-t", kind, "-w", "--stdin"], data=raw
                )
                .decode()
                .strip()
            )
            if actual != oid:
                raise Rejected("git_object_identity")
        self.git(self.root, ["update-ref", "HEAD", revision])
        self.git(self.root, ["read-tree", "HEAD"])

    def git(self, directory, args, data=None, limit=4194304):
        cmd = [
            self.gitbin,
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "diff.external=",
            "-c",
            "core.pager=cat",
            "-C",
            str(directory),
            *args,
        ]
        code, output, _ = self.run(cmd, data=data, limit=limit, env=ENV)
        if code:
            raise Rejected("git_operation_failed")
        return output

    def diff(self, files):
        if not isinstance(files, dict) or len(files) > 512:
            raise Rejected("snapshot_inventory")
        self.git(self.root, ["read-tree", "HEAD"])
        total = 0
        for path, item in files.items():
            if (
                not valid(path)
                or set(item) != {"mode", "data"}
                or item["mode"] not in ("100644", "100755")
            ):
                raise Rejected("unsafe_result_path")
            data = base64.b64decode(item["data"], validate=True)
            total += len(data)
            if len(data) > 262144 or total > 2097152:
                raise Rejected("result_snapshot_budget")
            oid = (
                self.git(
                    self.root,
                    ["hash-object", "--no-filters", "-w", "--stdin"],
                    data=data,
                )
                .decode()
                .strip()
            )
            self.git(
                self.root,
                ["update-index", "--add", "--cacheinfo", item["mode"], oid, path],
            )
        for path in self.base.keys() - files.keys():
            self.git(self.root, ["update-index", "--force-remove", "--", path])
        patch = self.git(
            self.root,
            [
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            ],
        )
        inventory = []
        for path in sorted(self.base.keys() | files.keys()):
            if self.base.get(path) == files.get(path):
                continue
            data = base64.b64decode(files[path]["data"]) if path in files else None
            inventory.append(
                {
                    "path": path,
                    "change": "added"
                    if path not in self.base
                    else "deleted"
                    if path not in files
                    else "modified",
                    "sha256": sha(data) if data is not None else None,
                    "bytes": len(data) if data is not None else 0,
                    "mode": files[path]["mode"] if path in files else None,
                }
            )
        return patch, inventory
