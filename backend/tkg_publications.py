"""Identity and affiliation rules shared by the release builder and research tools."""

import json
import os
import re
import unicodedata
from pathlib import Path

STATES = ("confirmed", "in_history", "unrecognised", "unknown")
STATE_RANK = {state: len(STATES) - i for i, state in enumerate(STATES)}
ASSERTED = {"verified_orcid", "self_reported"}
HIDDEN_AFFILIATION_SOURCE = "openalex_low_conf"
GENERIC = set('''university universite universitat universidad college institute institut institution
    school center centre hospital medical medicine health sciences science research department dept
    faculty laboratory lab national state of the and for at in a an system systems clinic
    foundation trust group division unit st saint'''.split())


def resolve_snapshot(directory: str) -> str:
    """An explicit, relocatable release pointer wins over legacy files in its parent."""
    base = Path(directory).resolve()
    pointer = base / "active-snapshot.json"
    if not pointer.is_file():
        return str(base)
    record = json.loads(pointer.read_text(encoding="utf-8"))
    target = (base / record["path"]).resolve()
    manifest = json.loads((target / "snapshot_manifest.json").read_text(encoding="utf-8"))
    if not record.get('snapshot_version') or manifest.get("snapshot_version") != record.get("snapshot_version"):
        raise RuntimeError("Active snapshot and manifest versions differ")
    return str(target)


def normalise_institution(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold().replace("&", " and ")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def institution_key(name: str, ror: str = "") -> str:
    return str(ror or "").strip() or normalise_institution(name) or "__none__"


def same_institution(name: str, ror: str, other_name: str, other_ror: str) -> bool:
    if ror and other_ror:
        return ror == other_ror
    a, b = normalise_institution(name), normalise_institution(other_name)
    if not a or not b:
        return False
    if a == b:
        return True
    left = {w for w in a.split() if w not in GENERIC and len(w) > 2}
    right = {w for w in b.split() if w not in GENERIC and len(w) > 2}
    small, large = sorted((left, right), key=len)
    return len(small) >= 2 and small <= large


def affiliation_state(institution: str, ror: str, history: list[dict]) -> str:
    if not str(institution or "").strip():
        return "unknown"
    matching = [row for row in history if same_institution(
        institution, ror, row.get("institution", ""), row.get("ror_id", ""))]
    if any(row.get("verification") in ASSERTED for row in matching):
        return "confirmed"
    return "in_history" if matching else "unrecognised"


def decisions_path() -> str:
    return os.environ.get("PROFILE_DECISIONS_DB") or str(
        Path(__file__).resolve().parents[2] / "bridge2aikg/work/state/publication-decisions.sqlite")


def displayable_affiliations(row: dict) -> list:
    """Affiliation rows shown to people: no low-confidence OpenAlex guesses or blank institutions."""
    return [
        a for a in (row.get("affiliations") or [])
        if a.get("source") != HIDDEN_AFFILIATION_SOURCE and (a.get("institution") or "").strip()
    ]
