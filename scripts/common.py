"""Shared helpers for the pipeline."""
import re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # .../paper_rerun
# inputs: ./in next to scripts/ if it exists, otherwise the folder one level up
IN = ROOT / "in" if (ROOT / "in").exists() else ROOT.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

def norm_name(s):
    """Upper-case, strip accents, unify separators (fixes 'Limbe' vs 'LIMBE' splits)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper().strip()
    s = re.sub(r"[_\-']+", " ", s)
    return re.sub(r"\s+", " ", s)
