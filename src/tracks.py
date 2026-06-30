"""
Competition tracks + per-track artifact namespacing.

A "track" is one modelling stream (men's NRL, NRLW, men's State of Origin,
women's State of Origin). The active track is chosen by the TRACK env var
(default "nrl"). Each track defines:

  - which Champion Data competitions feed it (include / exclude name regex),
  - the seasons used for training + out-of-time holdouts,
  - which track's *player-prop model* to predict with (Origin reuses the
    matching club model: soo -> nrl, soow -> nrlw).

All pipeline OUTPUTS are written under a per-track prefix:

    track "nrl"  -> data/processed/...      models/...      reports/...      reports/site/
    track "nrlw" -> data/processed/nrlw/... models/nrlw/... reports/nrlw/... reports/site/nrlw/

The default ("nrl") keeps the exact legacy paths, so the men's pipeline and the
nrl24-0.com fetch URLs are byte-for-byte unchanged.

data/raw is NOT namespaced: it is a shared match cache keyed by Champion Data
competition id, so men's / women's / Origin matches never collide.
"""
import os
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Track:
    name: str
    label: str
    include: re.Pattern          # competition name must match this
    exclude: re.Pattern | None   # ...and must NOT match this
    min_season: int              # earliest season with usable modern stats
    holdouts: tuple              # out-of-time validation seasons
    train_max: int               # last (complete) season to train through
    model_track: str             # which track's player-prop model to predict with
    # targets whose detailed stats only exist from `train_max` (e.g. NRLW run
    # metres / PCM first captured in 2025): trained on that one season only and
    # validated by within-season round holdout, then flagged "provisional".
    provisional_targets: frozenset = field(default_factory=frozenset)
    # the comp we PREDICT, when it's narrower than what we ingest. Origin tracks
    # ingest club+Origin (for player form) but only predict the Origin fixture.
    target_include: re.Pattern | None = None

    def matches(self, comp_name: str) -> bool:
        if not self.include.search(comp_name):
            return False
        if self.exclude is not None and self.exclude.search(comp_name):
            return False
        return True


TRACKS = {
    "nrl": Track(
        name="nrl", label="NRL",
        include=re.compile(r"\bNRL (Premiership|Finals)\b", re.I),
        exclude=re.compile(r"NRLW", re.I),
        min_season=2021, holdouts=(2023, 2024, 2025), train_max=2025,
        model_track="nrl",
    ),
    "nrlw": Track(
        name="nrlw", label="NRLW",
        include=re.compile(r"\bNRLW\b", re.I),          # "NRLW", "NRLW Finals", "2022B NRLW"
        exclude=re.compile(r"All Stars|Nines|Pacific", re.I),
        min_season=2022, holdouts=(2024, 2025), train_max=2025,
        model_track="nrlw",
        # Champion Data only began capturing NRLW run metres / PCM in 2025.
        provisional_targets=frozenset({"runMetres", "postContactMetres", "perf_points"}),
    ),
    # Origin tracks ingest BOTH the club comps AND the Origin comps, so every
    # representative player carries their club rolling-form history (Origin alone
    # is ~3 games/year). Prediction reuses the club model (model_track) and the
    # combined history; only the Origin fixture's squads/rosters differ.
    "soo": Track(
        name="soo", label="State of Origin",
        include=re.compile(r"\bNRL (Premiership|Finals)\b|State of Origin", re.I),
        exclude=re.compile(r"NRLW|Women", re.I),        # men's club + men's Origin
        min_season=2021, holdouts=(2024, 2025), train_max=2025,
        model_track="nrl",                              # predict with the men's club model
        target_include=re.compile(r"State of Origin", re.I),
    ),
    "soow": Track(
        name="soow", label="State of Origin (W)",
        include=re.compile(r"\bNRLW\b|State of Origin Women", re.I),
        exclude=re.compile(r"All Stars|Nines|Pacific", re.I),  # NRLW club + women's Origin
        min_season=2022, holdouts=(2024, 2025), train_max=2025,
        model_track="nrlw",                             # predict with the NRLW club model
        target_include=re.compile(r"State of Origin Women", re.I),
    ),
}


def track_name() -> str:
    return os.getenv("TRACK", "nrl")


def current() -> Track:
    name = track_name()
    if name not in TRACKS:
        raise SystemExit(f"unknown TRACK={name!r}; valid: {', '.join(TRACKS)}")
    return TRACKS[name]


# ---- competition selection -------------------------------------------------

def select_competitions(comps, track: Track | None = None):
    """Filter Champion Data's competition list to the active track, oldest first."""
    track = track or current()
    sel = [c for c in comps if track.matches(c["name"])]
    sel.sort(key=lambda c: (c["season"], c["id"]))
    return sel


# ---- per-track artifact paths ----------------------------------------------

def _prefix(track: Track | None = None) -> str:
    t = (track or current()).name
    return "" if t == "nrl" else t + "/"


def proc(fname: str = "", track: Track | None = None) -> str:
    return f"data/processed/{_prefix(track)}{fname}"


def model(fname: str = "", track: Track | None = None) -> str:
    return f"models/{_prefix(track)}{fname}"


def report(fname: str = "", track: Track | None = None) -> str:
    return f"reports/{_prefix(track)}{fname}"


def site_dir(track: Track | None = None) -> str:
    p = _prefix(track).rstrip("/")
    return f"reports/site/{p}" if p else "reports/site"


def model_for_prediction(fname: str, track: Track | None = None) -> str:
    """Player-prop model path, resolved to the track's model_track (Origin -> club)."""
    track = track or current()
    return model(fname, TRACKS[track.model_track])


RAW = "data/raw"   # shared, never namespaced (keyed by comp id)


def ensure_dirs(track: Track | None = None):
    for d in (proc("", track), model("", track), report("", track), site_dir(track)):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    t = current()
    print(f"active track: {t.name} ({t.label})")
    print(f"  include={t.include.pattern!r} exclude="
          f"{t.exclude.pattern if t.exclude else None!r}")
    print(f"  seasons>={t.min_season} holdouts={t.holdouts} model_track={t.model_track}")
    print(f"  proc={proc('')}  model={model('')}  report={report('')}  site={site_dir()}")
