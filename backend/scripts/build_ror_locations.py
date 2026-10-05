#!/usr/bin/env python3
"""Build the offline ROR -> location table that MATRIX Explore uses for geography filters.

The distinct ROR ids come from the active snapshot's papers.sqlite
(publication_groups.ror_id). Each is resolved to country, subdivision, city and
continent from the public ROR data dump (CC0), found through the Zenodo record
for the ROR data concept (6347574). Output:

    data/reference/ror_locations-<dump version>.json
        {"manifest": {...}, "locations": {"<9-character ROR id>": {...}}}

Usage:
    build_ror_locations.py                         # download the latest dump, build
    build_ror_locations.py --dump v2.13-...json    # use a dump already on disk
    build_ror_locations.py --source api            # fallback: polite ROR v2 API lookups

The API mode is for when the dump cannot be fetched. It is limited to about 5
requests per second, retries failures, and keeps a resumable cache file next to
the output (rerun it to continue).
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from geography import location_row, ror_key  # noqa: E402
from tkg_publications import resolve_snapshot  # noqa: E402

ZENODO_CONCEPT = "6347574"
ZENODO_QUERY = f"https://zenodo.org/api/records?q=conceptrecid:{ZENODO_CONCEPT}&sort=mostrecent&size=1"
USER_AGENT = "Bridge2AI-MATRIX/1.2 (ror-locations build)"
API_INTERVAL = 0.2  # seconds between API requests (5 per second at most)


def snapshot_rors(snapshot_dir: str) -> list[str]:
    db = sqlite3.connect(f"file:{Path(snapshot_dir) / 'papers.sqlite'}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT DISTINCT ror_id FROM publication_groups "
                          "WHERE ror_id IS NOT NULL AND ror_id != ''").fetchall()
    finally:
        db.close()
    return sorted({key for (value,) in rows if (key := ror_key(value))})


def fetch(url: str, timeout: int = 120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def download_dump(directory: Path) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    with fetch(ZENODO_QUERY, 60) as response:
        hit = json.load(response)["hits"]["hits"][0]
    entry = next(f for f in hit["files"] if f["key"].endswith(".zip"))
    target = directory / entry["key"]
    if not target.is_file() or target.stat().st_size != entry["size"]:
        with fetch(entry["links"]["self"], 600) as response, open(target, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    return target, {"zenodo_record": hit["id"], "version": hit["metadata"].get("version", ""),
                    "published": hit["metadata"].get("publication_date", ""),
                    "license": (hit["metadata"].get("license") or {}).get("id", ""),
                    "url": entry["links"]["self"], "md5": (entry.get("checksum") or "").removeprefix("md5:")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_dump(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".json"))
            with archive.open(name) as handle:
                return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_dump(wanted: list[str], records: list[dict]) -> dict[str, dict]:
    wanted_set, found = set(wanted), {}
    for record in records:
        key = ror_key(record.get("id"))
        if key in wanted_set and (row := location_row(record)):
            found[key] = row
    return found


def resolve_from_api(wanted: list[str], cache_path: Path) -> dict[str, dict | None]:
    """Resumable, rate-limited lookups. A null entry means the API returned no location."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    last, done = 0.0, 0
    for key in wanted:
        if key in cache:
            continue
        for attempt in range(4):
            wait = API_INTERVAL - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            try:
                with fetch(f"https://api.ror.org/v2/organizations/{key}", 30) as response:
                    cache[key] = location_row(json.load(response))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    cache[key] = None
                    break
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        done += 1
        if done % 200 == 0:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  {len(cache)}/{len(wanted)} looked up", flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {key: cache.get(key) for key in wanted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--source", choices=("dump", "api"), default="dump")
    parser.add_argument("--dump", help="ROR v2 dump (.json or the Zenodo .zip) already on disk")
    parser.add_argument("--source-url", default=f"https://doi.org/10.5281/zenodo.{ZENODO_CONCEPT}",
                        help="recorded in the manifest when --dump is used")
    parser.add_argument("--work-dir", default=os.environ.get("ROR_WORK_DIR", str(ROOT / "tmp" / "ror")))
    parser.add_argument("--output-dir", default=None, help="default: <data-dir>/reference")
    args = parser.parse_args()

    started = time.time()
    snapshot = Path(resolve_snapshot(args.data_dir))
    snapshot_version = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8")).get("snapshot_version", "")
    wanted = snapshot_rors(str(snapshot))
    print(f"{len(wanted)} distinct ROR ids in {snapshot.name} ({snapshot_version})")
    work = Path(args.work_dir)
    output_dir = Path(args.output_dir or Path(args.data_dir) / "reference")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "dump":
        if args.dump:
            dump, source = Path(args.dump), {}
            version = re.search(r"v\d+\.\d+", dump.name)
            source.update(version=version.group() if version else dump.stem, url=args.source_url, license="cc-zero")
        else:
            dump, source = download_dump(work)
        found = resolve_from_dump(wanted, read_dump(dump))
        manifest_source = {**source, "kind": "ror_data_dump", "file": dump.name, "sha256": sha256_file(dump)}
        version = source.get("version") or dump.stem
    else:
        found = {k: v for k, v in resolve_from_api(wanted, output_dir / ".ror_api_cache.json").items() if v}
        version = "api-" + datetime.now(timezone.utc).strftime("%Y%m%d")
        manifest_source = {"kind": "ror_v2_api", "url": "https://api.ror.org/v2/organizations/<id>", "license": "cc-zero"}

    unresolved = [key for key in wanted if key not in found]
    manifest = {
        "dump_version": version,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": manifest_source,
        "snapshot_version": snapshot_version,
        "ror_ids_requested": len(wanted),
        "resolved": len(found),
        "unresolved": len(unresolved),
        "unresolved_sample": unresolved[:20],
        "seconds": round(time.time() - started, 1),
    }
    path = output_dir / f"ror_locations-{version}.json"
    path.write_text(json.dumps({"manifest": manifest, "locations": found}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
