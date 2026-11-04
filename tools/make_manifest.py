"""Regenerate ck2/UPSTREAM_MANIFEST.json from a checkout of the upstream repo.

    git clone https://github.com/nikopueringer/CorridorKey2.0 /tmp/ck2
    git -C /tmp/ck2 checkout <commit>
    python tools/make_manifest.py /tmp/ck2

The manifest holds file names and SHA-256 digests only, never upstream code.
Bumping the pinned commit means re-running the parity test (tests/) as well.
"""

import hashlib
import json
import os
import subprocess
import sys

REPO = "nikopueringer/CorridorKey2.0"

# Everything the pack imports or reads at runtime. infer.py, the interactive
# launcher and the video encoders are deliberately left out.
INCLUDE_PREFIXES = ("corridorkey/", "third_party/MoGe/")
INCLUDE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "scripts/download_weights.py")


def main(src):
    commit = subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    tracked = subprocess.check_output(["git", "-C", src, "ls-files"], text=True).split("\n")
    files = {}
    for rel in sorted(t for t in tracked if t):
        if rel.startswith(INCLUDE_PREFIXES) or rel in INCLUDE_FILES:
            with open(os.path.join(src, rel), "rb") as f:
                files[rel] = hashlib.sha256(f.read()).hexdigest()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "ck2", "UPSTREAM_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"repo": REPO, "commit": commit, "files": files}, f, indent=1)
        f.write("\n")
    print(f"{len(files)} files @ {commit[:12]} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
