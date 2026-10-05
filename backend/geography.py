"""Offline geography for Explore: ROR id -> place, and matching of a typed place.

Geography here is the institution printed on a shared paper, resolved through its
ROR id. It is not a person's current residence. Locations come from a reference
table built from the public ROR data dump (scripts/build_ror_locations.py); a ROR
id that is missing from the table may use a bounded live ROR API lookup, whose
result (including a failure) is cached for the life of the process.
"""

import functools
import glob
import json
import logging
import os
import re
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROR_ID = re.compile(r"0[a-z0-9]{8}")
LIVE_LOOKUP_LIMIT = 40

# ISO 3166-1 alpha-2 -> alpha-3.
_ISO3 = dict(pair.split(":") for pair in """
AD:AND AE:ARE AF:AFG AG:ATG AI:AIA AL:ALB AM:ARM AO:AGO AQ:ATA AR:ARG AS:ASM AT:AUT AU:AUS AW:ABW AX:ALA AZ:AZE
BA:BIH BB:BRB BD:BGD BE:BEL BF:BFA BG:BGR BH:BHR BI:BDI BJ:BEN BL:BLM BM:BMU BN:BRN BO:BOL BQ:BES BR:BRA BS:BHS
BT:BTN BV:BVT BW:BWA BY:BLR BZ:BLZ CA:CAN CC:CCK CD:COD CF:CAF CG:COG CH:CHE CI:CIV CK:COK CL:CHL CM:CMR CN:CHN
CO:COL CR:CRI CU:CUB CV:CPV CW:CUW CX:CXR CY:CYP CZ:CZE DE:DEU DJ:DJI DK:DNK DM:DMA DO:DOM DZ:DZA EC:ECU EE:EST
EG:EGY EH:ESH ER:ERI ES:ESP ET:ETH FI:FIN FJ:FJI FK:FLK FM:FSM FO:FRO FR:FRA GA:GAB GB:GBR GD:GRD GE:GEO GF:GUF
GG:GGY GH:GHA GI:GIB GL:GRL GM:GMB GN:GIN GP:GLP GQ:GNQ GR:GRC GS:SGS GT:GTM GU:GUM GW:GNB GY:GUY HK:HKG HM:HMD
HN:HND HR:HRV HT:HTI HU:HUN ID:IDN IE:IRL IL:ISR IM:IMN IN:IND IO:IOT IQ:IRQ IR:IRN IS:ISL IT:ITA JE:JEY JM:JAM
JO:JOR JP:JPN KE:KEN KG:KGZ KH:KHM KI:KIR KM:COM KN:KNA KP:PRK KR:KOR KW:KWT KY:CYM KZ:KAZ LA:LAO LB:LBN LC:LCA
LI:LIE LK:LKA LR:LBR LS:LSO LT:LTU LU:LUX LV:LVA LY:LBY MA:MAR MC:MCO MD:MDA ME:MNE MF:MAF MG:MDG MH:MHL MK:MKD
ML:MLI MM:MMR MN:MNG MO:MAC MP:MNP MQ:MTQ MR:MRT MS:MSR MT:MLT MU:MUS MV:MDV MW:MWI MX:MEX MY:MYS MZ:MOZ NA:NAM
NC:NCL NE:NER NF:NFK NG:NGA NI:NIC NL:NLD NO:NOR NP:NPL NR:NRU NU:NIU NZ:NZL OM:OMN PA:PAN PE:PER PF:PYF PG:PNG
PH:PHL PK:PAK PL:POL PM:SPM PN:PCN PR:PRI PS:PSE PT:PRT PW:PLW PY:PRY QA:QAT RE:REU RO:ROU RS:SRB RU:RUS RW:RWA
SA:SAU SB:SLB SC:SYC SD:SDN SE:SWE SG:SGP SH:SHN SI:SVN SJ:SJM SK:SVK SL:SLE SM:SMR SN:SEN SO:SOM SR:SUR SS:SSD
ST:STP SV:SLV SX:SXM SY:SYR SZ:SWZ TC:TCA TD:TCD TF:ATF TG:TGO TH:THA TJ:TJK TK:TKL TL:TLS TM:TKM TN:TUN TO:TON
TR:TUR TT:TTO TV:TUV TW:TWN TZ:TZA UA:UKR UG:UGA UM:UMI US:USA UY:URY UZ:UZB VA:VAT VC:VCT VE:VEN VG:VGB VI:VIR
VN:VNM VU:VUT WF:WLF WS:WSM XK:XKX YE:YEM YT:MYT ZA:ZAF ZM:ZMB ZW:ZWE
""".split())

# Common spellings and aliases, already normalized, mapped to the ISO alpha-2 code.
_COUNTRY_ALIASES = {
    "us": "US", "usa": "US", "u s a": "US", "united states": "US", "united states of america": "US",
    "america": "US", "uk": "GB", "united kingdom": "GB", "great britain": "GB", "britain": "GB", "gb": "GB",
    "uae": "AE", "emirates": "AE", "south korea": "KR", "korea": "KR", "republic of korea": "KR",
    "north korea": "KP", "russia": "RU", "russian federation": "RU", "czech republic": "CZ", "czechia": "CZ",
    "holland": "NL", "netherlands": "NL", "ivory coast": "CI", "cote d ivoire": "CI", "iran": "IR",
    "taiwan": "TW", "vietnam": "VN", "viet nam": "VN", "turkey": "TR", "turkiye": "TR", "hong kong": "HK",
    "macau": "MO", "macao": "MO", "burma": "MM", "swaziland": "SZ", "eswatini": "SZ", "macedonia": "MK",
    "north macedonia": "MK", "laos": "LA", "syria": "SY", "palestine": "PS", "dr congo": "CD",
    "democratic republic of the congo": "CD", "congo kinshasa": "CD", "gambia": "GM", "bolivia": "BO",
    "venezuela": "VE", "tanzania": "TZ", "moldova": "MD", "brunei": "BN", "cape verde": "CV",
    "east timor": "TL", "timor leste": "TL", "vatican": "VA", "puerto rico": "PR",
}


def normalize(value) -> str:
    """Trim, casefold, drop accents and punctuation ("U.S." -> "us"), collapse spaces."""
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", text)  # u.s.a -> usa
    text = text.replace(".", "").replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def ror_key(value) -> str:
    key = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1].casefold()
    return key if ROR_ID.fullmatch(key) else ""


def location_row(record: dict) -> dict | None:
    """The first listed location of a ROR v2 record, in the reference-table shape."""
    for item in (record or {}).get("locations") or []:
        details = item.get("geonames_details") if isinstance(item, dict) else None
        if isinstance(details, dict) and details.get("country_code"):
            return {
                "country_code": details.get("country_code") or "",
                "country_name": details.get("country_name") or "",
                "subdivision_code": details.get("country_subdivision_code") or "",
                "subdivision_name": details.get("country_subdivision_name") or "",
                "city": details.get("name") or "",
                "continent_code": details.get("continent_code") or "",
                "continent_name": details.get("continent_name") or "",
                "lat": details.get("lat"),
                "lng": details.get("lng"),
            }
    return None


def _reference_path() -> str | None:
    configured = os.environ.get("ROR_LOCATIONS_PATH", "").strip()
    if configured:
        return configured if os.path.isfile(configured) else None
    found = sorted(glob.glob(os.path.join(_ROOT, "data", "reference", "ror_locations-*.json")),
                   key=lambda path: (os.path.getmtime(path), path))
    return found[-1] if found else None


def _live_lookup(key: str) -> dict | None:
    request = urllib.request.Request(f"https://api.ror.org/v2/organizations/{key}",
                                     headers={"User-Agent": "Bridge2AI-MATRIX/1.2"})
    with urllib.request.urlopen(request, timeout=6) as response:
        record = json.load(response)
    return location_row(record) if record.get("id") == f"https://ror.org/{key}" else None


class Geography:
    """Resolves ROR ids to places and tests places against a typed filter."""

    def __init__(self, table: dict | None = None, live=_live_lookup, version: str = ""):
        self.table = table or {}
        self.version = version
        self._live = live
        self._cache = {}  # key -> row or None; failures are kept for the process lifetime
        self.live_lookups = 0

    @classmethod
    def load(cls, path: str | None = None, live=_live_lookup) -> "Geography":
        path = path or _reference_path()
        if not path:
            logger.warning("No ROR locations table found; geography filters use live ROR lookups only")
            return cls({}, live)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        version = (data.get("manifest") or {}).get("dump_version", "")
        logger.info("ROR locations loaded: %s (%d institutions)", version or path, len(data.get("locations") or {}))
        return cls(data.get("locations") or {}, live, version)

    def resolve(self, rors) -> tuple[dict, bool]:
        """(key -> location row, complete). Unknown ids use at most LIVE_LOOKUP_LIMIT live lookups."""
        found, pending = {}, []
        for ror in dict.fromkeys(rors):
            key = ror_key(ror)
            if not key:
                continue
            if key in self.table:
                found[key] = self.table[key]
            elif key in self._cache:
                if self._cache[key]:
                    found[key] = self._cache[key]
            else:
                pending.append(key)
        complete = True
        if pending and self._live is not None:
            complete = len(pending) <= LIVE_LOOKUP_LIMIT

            def read(key):
                try:
                    return key, self._live(key)
                except Exception:
                    return key, None

            with ThreadPoolExecutor(max_workers=4) as pool:
                for key, row in pool.map(read, pending[:LIVE_LOOKUP_LIMIT]):
                    self._cache[key] = row
                    self.live_lookups += 1
                    if row:
                        found[key] = row
        return found, complete

    # ----- matching -----

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _keys(code, name, sub_code, sub_name, city, continent) -> frozenset:
        """(field, normalized value) pairs under which one location can be found."""
        keys = set()
        code = (code or "").upper()
        if code:
            keys.add(("country_code", code.casefold()))
            if code in _ISO3:
                keys.add(("country_code3", _ISO3[code].casefold()))
            keys.add(("country", normalize(name)))
            keys.add(("country", normalize(re.sub(r"^the\s+", "", name or "", flags=re.I))))
            keys.update(("country", alias) for alias, target in _COUNTRY_ALIASES.items() if target == code)
        if sub_name:
            keys.add(("subdivision", normalize(sub_name)))
        if sub_code and code in {"US", "CA"}:
            keys.add(("subdivision_code", normalize(sub_code)))
            keys.add(("subdivision_code", normalize(f"{code}-{sub_code}")))
        if city:
            keys.add(("city", normalize(city)))
        if continent:
            keys.add(("continent", normalize(continent)))
        return frozenset(keys)

    @staticmethod
    def parse(query) -> list[str]:
        """A typed place as normalized parts. "Boston, MA" is two parts that must both match."""
        pieces = (re.sub(r"^the ", "", normalize(piece)) for piece in re.split(r"[;,]", str(query or "")))
        return [piece for piece in pieces if piece]

    def match(self, row: dict, parts: list[str]) -> list[dict] | None:
        """The (term, field) matches if every part matches this location, else None."""
        keys = self._keys(row.get("country_code"), row.get("country_name"), row.get("subdivision_code"),
                          row.get("subdivision_name"), row.get("city"), row.get("continent_name"))
        by_value = {}
        for field, value in keys:
            by_value.setdefault(value, []).append(field)
        matched = []
        for part in parts:
            fields = by_value.get(part, [])
            if len(parts) == 1 and len(part) == 2 and part.upper() in _ISO3:
                # Alone, a two-letter code is a country (CA is Canada); use US-CA or California for the state.
                fields = [f for f in fields if f == "country_code"]
            if not fields:
                return None
            matched.extend({"term": part, "field": field} for field in fields)
        return matched


@functools.lru_cache(maxsize=1)
def default() -> Geography:
    return Geography.load()
