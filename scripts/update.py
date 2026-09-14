import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
YEAR = int(CFG["season"])

IDP_POSITIONS = {"DT", "DE", "LB", "CB", "S", "DB", "DL"}


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def clean_id(value):
    if value is None:
        return ""
    s = str(value).strip()
    return re.sub(r"\.0$", "", s)


def norm(value):
    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def export_base(league_url):
    p = urlparse(league_url)
    # MFL's export API is served from the league host and season path.
    return f"{p.scheme}://{p.netloc}/{YEAR}/export"


def mfl_export(league, export_type, **extra):
    params = {"TYPE": export_type, "L": league["league_id"], "JSON": 1, **extra}
    r = requests.get(
        export_base(league["url"]),
        params=params,
        timeout=45,
        headers={"User-Agent": "nfl-dynasty-snap-dashboard/1.0"},
    )
    r.raise_for_status()
    return r.json()


def mfl_roster(league):
    payload = mfl_export(league, "rosters", FRANCHISE=league["franchise_id"])
    franchises = as_list(payload.get("rosters", {}).get("franchise"))
    if not franchises:
        raise RuntimeError(f"No roster returned for {league['name']}")
    franchise = franchises[0]
    out = []
    for p in as_list(franchise.get("player")):
        if isinstance(p, str):
            out.append({"id": clean_id(p), "status": "ROSTER"})
        else:
            out.append({"id": clean_id(p.get("id")), "status": p.get("status", "ROSTER")})
    return [x for x in out if x["id"]]


def mfl_players(league):
    payload = mfl_export(league, "players", DETAILS=1)
    players = as_list(payload.get("players", {}).get("player"))
    out = {}
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = clean_id(p.get("id"))
        if pid:
            out[pid] = p
    return out


# 1. Pull current ownership from MFL.
roster_records = []
mfl_player_details = {}
for league in CFG["leagues"]:
    details = mfl_players(league)
    mfl_player_details.update(details)
    for player in mfl_roster(league):
        roster_records.append({**player, "league": league["name"]})

# 2. Pull the stable MFL -> PFR crosswalk and NFL snap counts.
ff_ids = nfl.load_ff_playerids().to_pandas()
snaps = nfl.load_snap_counts(seasons=[YEAR]).to_pandas()
ff_ids.columns = [str(c) for c in ff_ids.columns]
snaps.columns = [str(c) for c in snaps.columns]

mfl_col = next(c for c in ff_ids.columns if c.lower() == "mfl_id")
pfr_col = next(c for c in ff_ids.columns if c.lower() in {"pfr_id", "pfr_player_id"})
name_col = next(c for c in ff_ids.columns if c.lower() == "name")
pos_col = next((c for c in ff_ids.columns if c.lower() == "position"), None)

ff_ids["mfl_key"] = ff_ids[mfl_col].map(clean_id)
ff_ids["pfr_key"] = ff_ids[pfr_col].fillna("").map(clean_id)
ff_ids = ff_ids[ff_ids["mfl_key"].ne("")].drop_duplicates("mfl_key", keep="first")

snaps["week_num"] = pd.to_numeric(snaps["week"], errors="coerce")
snaps = snaps[snaps["game_type"].astype(str).str.upper().eq("REG")].copy()
snaps["pfr_player_id"] = snaps["pfr_player_id"].fillna("").map(clean_id)

# Snap counts are game-level. A player can have one row per game in a week,
# so aggregate to weekly totals and calculate the weekly percentage from the
# available game rows (weighted by the NFL snap percentages' underlying snaps).
num_cols = ["offense_snaps", "defense_snaps", "st_snaps"]
for col in num_cols:
    snaps[col] = pd.to_numeric(snaps[col], errors="coerce").fillna(0)

weekly = (
    snaps.groupby(["pfr_player_id", "week_num", "player", "position", "team"], as_index=False)[num_cols]
    .sum()
)

# NFL team snap totals can vary by game, so derive percentage from each game's
# percentage weighted by the team's snap count. For dashboard purposes the
# summed snap count divided by the summed estimated team snaps is equivalent.
def weighted_pct(group, snap_col, pct_col):
    s = pd.to_numeric(group[snap_col], errors="coerce").fillna(0).sum()
    pct = pd.to_numeric(group[pct_col], errors="coerce")
    snaps_taken = pd.to_numeric(group[snap_col], errors="coerce").fillna(0)

    # nflverse has used both fractional (0-1) and percentage (0-100)
    # representations for snap percentages. Normalize to a fraction before
    # recovering the team snap denominator, then return a display percentage.
    valid = pct.notna() & (pct > 0) & (snaps_taken > 0)
    if not valid.any():
        return None
    frac = pct.where(pct <= 1, pct / 100)
    denoms = (snaps_taken / frac.where(frac > 0)).where(valid).dropna()
    denom = denoms.sum() if len(denoms) else 0
    return (s / denom * 100) if denom else None

pct_rows = []
for keys, g in snaps.groupby(["pfr_player_id", "week_num"]):
    row = {"pfr_player_id": keys[0], "week_num": int(keys[1])}
    for snap_col, pct_col, out_col in [
        ("offense_snaps", "offense_pct", "offense_pct"),
        ("defense_snaps", "defense_pct", "defense_pct"),
        ("st_snaps", "st_pct", "st_pct"),
    ]:
        row[out_col] = weighted_pct(g, snap_col, pct_col)
    pct_rows.append(row)
pcts = pd.DataFrame(pct_rows)
weekly = weekly.merge(pcts, on=["pfr_player_id", "week_num"], how="left")

latest_week = int(weekly["week_num"].max()) if len(weekly) else 0

# 3. Consolidate duplicate ownership by stable MFL player ID.
roster_df = pd.DataFrame(roster_records)
roster_df["mfl_key"] = roster_df["id"].map(clean_id)
roster_df = roster_df.merge(
    ff_ids[["mfl_key", "pfr_key", name_col] + ([pos_col] if pos_col else [])],
    on="mfl_key", how="left"
)

rows = {}
for _, r in roster_df.iterrows():
    key = r["mfl_key"]
    detail = mfl_player_details.get(key, {})
    name = r.get(name_col) if pd.notna(r.get(name_col)) else detail.get("name")
    if not name:
        name = f"MFL {key}"
    position = r.get(pos_col) if pos_col and pd.notna(r.get(pos_col)) else detail.get("position", "")
    position = str(position or "")
    unit = "IDP" if position.upper() in IDP_POSITIONS else "Offense"
    if key not in rows:
        rows[key] = {
            "player_id": key,
            "player": str(name),
            "position": position,
            "unit": unit,
            "team": "",
            "leagues": [],
            "status": [],
            "pfr_id": clean_id(r.get("pfr_key", "")),
        }
    if r["league"] not in rows[key]["leagues"]:
        rows[key]["leagues"].append(r["league"])
    rows[key]["status"].append({"league": r["league"], "status": str(r["status"])})

# 4. Build player history and current-week metrics.
history = {}
for _, r in weekly.iterrows():
    pid = clean_id(r["pfr_player_id"])
    if not pid:
        continue
    history.setdefault(pid, []).append({
        "week": int(r["week_num"]),
        "team": str(r["team"]),
        "offense_snaps": int(r["offense_snaps"]),
        "offense_pct": None if pd.isna(r["offense_pct"]) else round(float(r["offense_pct"]), 2),
        "defense_snaps": int(r["defense_snaps"]),
        "defense_pct": None if pd.isna(r["defense_pct"]) else round(float(r["defense_pct"]), 2),
        "st_snaps": int(r["st_snaps"]),
        "st_pct": None if pd.isna(r["st_pct"]) else round(float(r["st_pct"]), 2),
    })

for player in rows.values():
    pid = player["pfr_id"]
    hist = sorted(history.get(pid, []), key=lambda x: x["week"])
    metric = "defense" if player["unit"] == "IDP" else "offense"
    snap_key = f"{metric}_snaps"
    pct_key = f"{metric}_pct"
    recent = [x for x in hist if x["week"] <= latest_week]
    last = recent[-1] if recent else None
    if last:
        player["team"] = last["team"]
        player["snaps"] = last[snap_key]
        player["pct"] = last[pct_key]
        vals = [x[pct_key] for x in recent[-3:] if x[pct_key] is not None]
        all_vals = [x[pct_key] for x in recent if x[pct_key] is not None]
        player["avg_3wk"] = round(sum(vals) / len(vals), 1) if vals else None
        player["season_avg"] = round(sum(all_vals) / len(all_vals), 1) if all_vals else None
        if len(recent) >= 2 and recent[-2][pct_key] is not None and last[pct_key] is not None:
            delta = last[pct_key] - recent[-2][pct_key]
            player["delta"] = round(delta, 1)
            player["trend"] = 1 if delta >= 5 else (-1 if delta <= -5 else 0)
        else:
            player["delta"] = None
            player["trend"] = 0
    else:
        player["snaps"] = None
        player["pct"] = None
        player["avg_3wk"] = None
        player["season_avg"] = None
        player["delta"] = None
        player["trend"] = 0
    player["history"] = hist

payload = {
    "season": YEAR,
    "week": latest_week,
    "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "status": "Live data",
    "leagues": [x["name"] for x in CFG["leagues"]],
    "current": list(rows.values()),
}
(ROOT / "data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Updated {len(rows)} unique current players across {len(CFG['leagues'])} leagues; latest week={latest_week}")
