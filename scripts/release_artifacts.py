#!/usr/bin/env python3
"""Package, verify and restore the per-run artifacts that git does not carry.

Every confirmation-protocol analysis reads two files per run, and `*.npz` is
excluded from the repository because together they are about 2.2 GB:

    artifacts/test_prefix.npz   test row indices, true labels, predictions and
                                the cumulative output-spike readout per timestep
    artifacts/val_scores.npz    validation labels and attack scores

They exist for 960 runs: the confirmation sweep (results/runs/) and the post-hoc
E1, A2, A2-T and E3 arms (results/runs_e1/, runs_a2/, runs_a2t/, runs_e3/), so
1,920 files. The npz files are already compressed, so the archive is a plain tar.

    python scripts/release_artifacts.py build     # archive + checksum list
    python scripts/release_artifacts.py verify    # files on disk vs the list
    python scripts/release_artifacts.py fetch     # download, verify, extract

`build` writes the archive to results/cache/ (gitignored) and two small files
that are committed:

    results/ARTIFACTS.sha256   one line per file in `sha256sum` format, so
                               `sha256sum -c results/ARTIFACTS.sha256` also works
                               from this directory
    results/ARTIFACTS.json     the archive's name, size and SHA-256, and the URL
                               and DOI it is published under once it is

The archive is deterministic (members sorted; timestamps, ownership and modes
fixed), so rebuilding it from the same files gives the same bytes and hash.

`fetch` takes the URL from results/ARTIFACTS.json unless --url or a local
--archive is given. It checks the archive's SHA-256 before opening it, extracts
only the paths the checksum list names, never overwrites a file whose content
differs, and finishes with `verify`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_TREES = ["results/runs", "results/runs_e1", "results/runs_a2",
             "results/runs_a2t", "results/runs_e3"]
NAMES = ("test_prefix.npz", "val_scores.npz")
SUMS = ROOT / "results/ARTIFACTS.sha256"
META = ROOT / "results/ARTIFACTS.json"
CACHE = ROOT / "results/cache"
ARCHIVE = "intrasnneval-run-artifacts.tar"


def artifact_paths(root: Path = ROOT) -> list[str]:
    """Every artifact a completed run should have, relative to `root`."""
    out = []
    for tree in RUN_TREES:
        for rj in (root / tree).rglob("results.json"):
            out += [str((rj.parent / "artifacts" / n).relative_to(root))
                    for n in NAMES]
    return sorted(out)


def missing_artifacts(root: Path = ROOT) -> list[str]:
    return [p for p in artifact_paths(root) if not (root / p).exists()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums() -> dict[str, str]:
    if not SUMS.exists():
        sys.exit(f"{SUMS.relative_to(ROOT)} not found; run `build` first")
    pairs = (line.split("  ", 1) for line in SUMS.read_text().splitlines())
    return {path: digest for digest, path in pairs}


def build() -> int:
    paths = artifact_paths()
    missing = missing_artifacts()
    if missing:
        sys.exit(f"{len(missing)} of {len(paths)} artifacts are missing, e.g. "
                 f"{missing[0]}; refusing to package an incomplete set")

    sums = {p: sha256(ROOT / p) for p in paths}
    SUMS.write_text("".join(f"{sums[p]}  {p}\n" for p in paths))

    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / ARCHIVE
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as tar:
        for p in paths:
            info = tar.gettarinfo(ROOT / p, arcname=p)
            info.mtime, info.mode = 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(ROOT / p, "rb") as f:
                tar.addfile(info, f)

    meta = json.loads(META.read_text()) if META.exists() else {}
    meta.update({"archive": ARCHIVE, "bytes": archive.stat().st_size,
                 "sha256": sha256(archive), "files": len(paths),
                 "checksums": str(SUMS.relative_to(ROOT))})
    meta.setdefault("url", None)
    meta.setdefault("doi", None)
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(paths)} files -> {archive} "
          f"({meta['bytes'] / 1e9:.2f} GB, sha256 {meta['sha256'][:16]}...)")
    print(f"checksums -> {SUMS.relative_to(ROOT)}; "
          f"archive record -> {META.relative_to(ROOT)}")
    return 0


def verify(root: Path = ROOT) -> int:
    sums = read_sums()
    missing = [p for p in sums if not (root / p).exists()]
    bad = [p for p in sums
           if (root / p).exists() and sha256(root / p) != sums[p]]
    unlisted = sorted(set(artifact_paths(root)) - set(sums))
    print(f"{len(sums) - len(missing) - len(bad)} of {len(sums)} artifacts "
          f"present and matching under {root}")
    for label, items in (("missing", missing), ("CHECKSUM MISMATCH", bad),
                         ("run with no listed artifact", unlisted)):
        if items:
            print(f"  {len(items)} {label}, e.g. {items[0]}")
    return 1 if (missing or bad or unlisted) else 0


def fetch(url: str | None, archive: Path | None, dest: Path) -> int:
    meta = json.loads(META.read_text()) if META.exists() else {}
    if archive is None:
        url = url or meta.get("url")
        if not url:
            sys.exit("no download URL: the archive is not published yet. Pass "
                     "--url, or set `url` in results/ARTIFACTS.json")
        CACHE.mkdir(parents=True, exist_ok=True)
        archive = CACHE / meta.get("archive", ARCHIVE)
        part = archive.with_suffix(archive.suffix + ".part")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, part)
        part.replace(archive)

    if meta.get("sha256") and sha256(archive) != meta["sha256"]:
        sys.exit(f"{archive} does not match the SHA-256 in "
                 f"results/ARTIFACTS.json; not extracting it")

    sums = read_sums()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        stray = [m.name for m in members if m.name not in sums or not m.isfile()]
        if stray:
            sys.exit(f"archive holds {len(stray)} unlisted entries, e.g. "
                     f"{stray[0]}; not extracting it")
        clash = [m.name for m in members if (dest / m.name).exists()
                 and sha256(dest / m.name) != sums[m.name]]
        if clash:
            sys.exit(f"{len(clash)} files already exist with different content, "
                     f"e.g. {clash[0]}; move them aside and fetch again")
        todo = [m for m in members if not (dest / m.name).exists()]
        for m in todo:
            tar.extract(m, dest, filter="data")
    print(f"extracted {len(todo)} files into {dest} "
          f"({len(members) - len(todo)} already present)")
    return verify(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    v = sub.add_parser("verify")
    v.add_argument("--dest", type=Path, default=ROOT)
    f = sub.add_parser("fetch")
    f.add_argument("--url")
    f.add_argument("--archive", type=Path)
    f.add_argument("--dest", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.cmd == "build":
        return build()
    if args.cmd == "verify":
        return verify(args.dest)
    return fetch(args.url, args.archive, args.dest)


if __name__ == "__main__":
    sys.exit(main())
