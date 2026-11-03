"""Trusted collector/initializer. No repository code is imported or executed."""

import base64, json, os, re, stat, sys
from pathlib import Path

ROOT = Path("/workspace")


def valid(path):
    parts = path.split("/")
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
            for p in parts
        )
    )


def collect():
    result = {}
    total = 0

    def visit(directory, prefix=""):
        nonlocal total
        for entry in os.scandir(directory):
            path = prefix + entry.name
            if not valid(path):
                raise ValueError("unsafe path")
            s = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(s.st_mode):
                visit(entry.path, path + "/")
            elif stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= 262144:
                total += s.st_size
                if total > 2097152 or len(result) >= 512:
                    raise ValueError("snapshot bound")
                fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    data = os.read(fd, 262145)
                finally:
                    os.close(fd)
                if len(data) != s.st_size:
                    raise ValueError("unstable file")
                result[path] = {
                    "mode": "100755" if s.st_mode & 0o111 else "100644",
                    "data": base64.b64encode(data).decode(),
                }
            else:
                raise ValueError("links/special/oversized files forbidden")

    visit(ROOT)
    return result


if __name__ == "__main__":
    try:
        if sys.argv[1] == "init":
            files = json.loads(Path("/config/base.json").read_text())
            for path, value in files.items():
                if not valid(path):
                    raise ValueError("unsafe path")
                target = ROOT / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(value["data"], validate=True))
                target.chmod(0o755 if value["mode"] == "100755" else 0o644)
            collect()
        else:
            print(json.dumps(collect(), sort_keys=True))
    except Exception:
        print("Unsafe or over-budget workspace", file=sys.stderr)
        sys.exit(1)
