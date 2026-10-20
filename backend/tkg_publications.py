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
# Keep institution matching identical to Graph's lib/affiliationPolicy.mjs.
# Removing additional words such as "clinic" or "system" changes ownership
# eligibility (Mayo Clinic versus UC San Diego Health System, for example).
GENERIC = set('''university college institute institution school center centre hospital
    medical medicine health sciences science research department faculty laboratory lab
    national state of the and for at in'''.split())


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


def normalise_institution(value: str, *, expand_ampersand: bool = True) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    if expand_ampersand:
        text = text.replace("&", " and ")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def institution_key(name: str, ror: str = "") -> str:
    return str(ror or "").strip() or normalise_institution(name) or "__none__"


def same_institution(name: str, ror: str, other_name: str, other_ror: str) -> bool:
    ror, other_ror = str(ror or '').strip(), str(other_ror or '').strip()
    if ror and other_ror:
        return re.sub(r'^https?://ror.org/', '', ror) == re.sub(r'^https?://ror.org/', '', other_ror)
    a = normalise_institution(name, expand_ampersand=False)
    b = normalise_institution(other_name, expand_ampersand=False)
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


def displayable_affiliations(row: dict, reviews=()) -> list:
    """Actual ORCID records take precedence; approved corrections remain separate."""
    source = [a for a in row.get('affiliations', [])
              if a.get('source') != HIDDEN_AFFILIATION_SOURCE and str(a.get('institution') or '').strip()]
    by_key = {r['group_id']: r for r in reviews}
    has_orcid = any(a.get('source') == 'orcid' for a in source)
    selected = {}

    def retain(row, key, override, trusted):
        fields = {k: v for k, v in (override or {}).items() if k != 'records'}
        for record in (override or {}).get('records', [{}]):
            value = {**row, **fields, **record, 'group_id': key, 'trusted': trusted}
            if override:
                value['source'] = 'owner_approved'
            selected[(key, value.get('role_title'), value.get('start_year'), value.get('end_year'))] = value

    for a in source:
        key = institution_key(a.get('institution'), a.get('ror_id'))
        review = by_key.get(key, {})
        if review.get('state') == 'rejected': continue
        if has_orcid and a.get('source') != 'orcid' and review.get('state') != 'approved': continue
        override = review.get('override') if review.get('state') == 'approved' else None
        retain(a, key, override, a.get('source') == 'orcid' or review.get('state') == 'approved')
    source_keys = {institution_key(a.get('institution'), a.get('ror_id')) for a in source}
    for review in reviews:
        if review.get('state') == 'approved' and review.get('override') and review['group_id'] not in source_keys:
            retain({}, review['group_id'], review['override'], True)
    return list(selected.values())


def primary_affiliation(rows: list[dict]) -> str:
    """Match the Graph header: one current trusted institution, else latest."""
    rows = [r for r in rows if str(r.get('institution') or '').strip()]
    trusted = [r for r in rows if r.get('trusted') or r.get('source') in {'orcid', 'owner_approved'}]
    ordered = sorted(trusted or rows,
                     key=lambda r: (bool(r.get('is_current')), r.get('end_year') or r.get('start_year') or 0),
                     reverse=True)
    current = [r for r in ordered if r.get('is_current') and not r.get('end_year')]
    latest = sorted(ordered, key=lambda r: r.get('end_year') or r.get('start_year') or 0, reverse=True)
    selected = current or latest
    return selected[0]['institution'] if selected else ''
