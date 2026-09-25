"""Apply _brand.yml to the Streamlit app and its Plotly charts.

Streamlit does not read _brand.yml itself, so this module:
* resolves brand colours and fonts from _brand.yml,
* writes the matching .streamlit/config.toml theme (run ``uv run python brand.py``),
* provides CSS and a Plotly template for the app.

Channel colours are a validated colour-blind-safe categorical palette in a fixed
order, so a channel keeps its colour on every chart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import yaml

BRAND_FILE = Path(__file__).resolve().parent / "_brand.yml"

# Built from Posit and Snowflake brand hues, adjusted where needed to pass the
# colour-blind checks (dataviz validate_palette.js, light mode on white).
CHANNEL_COLORS = {
    "Paid Search": "#1F6FB5",    # Posit / Snowflake blue family
    "Connected TV": "#EE6331",   # Posit orange
    "Paid Social": "#29B5E8",    # Snowflake blue
    "Display": "#72994E",        # Posit green
    "Affiliate": "#7D44CF",      # Snowflake purple
    "Email / CRM": "#D98A00",    # Snowflake amber, darkened
    "Direct Mail": "#9A4665",    # Posit burgundy
    "Podcast": "#00A19B",        # teal
    "Baseline": "#C8D1D8",
}


@lru_cache
def brand() -> dict:
    """Brand colours (resolved to hex) and font names."""
    raw = yaml.safe_load(BRAND_FILE.read_text())
    palette = raw["color"]["palette"]
    colors = {k: palette.get(v, v) for k, v in raw["color"].items() if k != "palette"}
    colors.update(palette)
    typo = raw["typography"]
    weights = {f["family"]: f.get("weight", [400]) for f in typo["fonts"]}
    return dict(
        name=raw["meta"]["name"]["short"],
        colors=colors,
        base_font=typo["base"]["family"],
        heading_font=typo["headings"]["family"],
        weights=weights,
    )


def font_url(family: str) -> str:
    """Google Fonts CSS URL for a brand font, with the weights listed in _brand.yml."""
    weights = ";".join(str(w) for w in sorted(brand()["weights"].get(family, [400])))
    return f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:wght@{weights}&display=swap"


def css() -> str:
    b = brand()
    c = b["colors"]
    return f"""
<style>
@import url('{font_url(b['heading_font'])}');
@import url('{font_url(b['base_font'])}');
h1, h2, h3, h4 {{ font-family: '{b['heading_font']}', sans-serif !important; color: {c['foreground']}; }}
h1 {{ font-weight: 700 !important; letter-spacing: -.01em; }}
[data-testid="stMetric"] {{ background: {c['white']}; border: 1px solid {c['line']};
    border-radius: 10px; padding: 14px 16px; }}
[data-testid="stMetricValue"] {{ font-family: '{b['heading_font']}', sans-serif; font-weight: 700; font-size: 2rem; }}
[data-testid="stMetricLabel"] p {{ color: {c['secondary']}; text-transform: uppercase;
    letter-spacing: .06em; font-size: .72rem; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1rem; }}
.lv-topbar {{ height: 4px; border-radius: 2px; margin: -.5rem 0 1rem;
    background: linear-gradient(90deg, {c['posit-blue']} 0%, {c['posit-blue']} 50%, {c['snowflake-blue']} 50%); }}
.lv-eyebrow {{ color: {c['secondary']}; font-size: .8rem; font-weight: 600; margin-bottom: -.4rem; }}
.lv-disclaimer {{ color: {c['secondary']}; font-size: .78rem; border-top: 1px solid {c['line']};
    padding-top: .6rem; margin-top: 1.5rem; }}
.lv-callout {{ background: {c['well']}; border-left: 4px solid {c['snowflake-blue']};
    padding: .7rem 1rem; border-radius: 6px; margin: .4rem 0 1rem; }}
.lv-source {{ display:inline-block; font-size:.75rem; padding:.15rem .6rem; border-radius:99px;
    background:{c['white']}; border:1px solid {c['line']}; color:{c['secondary']}; }}
.lv-badge {{ display:inline-block; font-size:.8rem; font-weight:700; padding:.2rem .7rem;
    border-radius:99px; background:{c['mid-blue']}; color:{c['white']}; letter-spacing:.02em; }}
.lv-section {{ color: {c['secondary']}; text-transform: uppercase; letter-spacing: .1em;
    font-size: .72rem; font-weight: 700; margin: .2rem 0 .1rem; }}
.lv-headline {{ font-family: '{b['heading_font']}', sans-serif; font-size: 1.45rem; font-weight: 700;
    color: {c['foreground']}; margin: 0 0 .6rem; }}
.lv-headline b {{ color: {c['posit-orange']}; }}
.lv-alloc {{ display:grid; grid-template-columns: 8.5rem 1fr 5.5rem; align-items:center;
    gap:.6rem; margin:.28rem 0; font-size:.9rem; color:{c['foreground']}; }}
.lv-alloc .bar {{ height: 12px; border-radius: 0 4px 4px 0; min-width: 2px; }}
.lv-alloc .track {{ background:{c['well']}; border-radius: 4px; }}
.lv-alloc .val {{ text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }}
.lv-alloc .val.zero {{ color:{c['secondary']}; font-weight:400; }}
.lv-kv {{ color:{c['secondary']}; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }}
.lv-kv-value {{ font-family: '{b['heading_font']}', sans-serif; font-size:1.2rem; font-weight:700;
    color:{c['foreground']}; }}
.lv-note {{ color:{c['secondary']}; font-size:.85rem; }}
.lv-tag {{ display:inline-block; font-size:.7rem; padding:.05rem .5rem; border-radius:4px;
    border:1px dashed {c['secondary']}; color:{c['secondary']}; text-transform:uppercase;
    letter-spacing:.08em; }}
</style>
"""


def plotly_template() -> go.layout.Template:
    b = brand()
    c = b["colors"]
    t = go.layout.Template(pio.templates["plotly_white"])
    t.layout.update(
        font=dict(family=b["base_font"], color=c["foreground"], size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(CHANNEL_COLORS.values()),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font_family=b["base_font"], bgcolor=c["white"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        title=dict(font=dict(family=b["heading_font"], size=18)),
    )
    t.layout.xaxis.update(gridcolor="#E8EEF2", zeroline=False, linecolor=c["line"])
    t.layout.yaxis.update(gridcolor="#E8EEF2", zeroline=False, linecolor=c["line"])
    return t


def write_streamlit_config() -> Path:
    """Write .streamlit/config.toml so Streamlit's own widgets use the brand theme."""
    b = brand()
    c = b["colors"]
    path = BRAND_FILE.parent / ".streamlit" / "config.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"""# Generated from _brand.yml by `uv run python brand.py`. Edit _brand.yml, not this file.
[theme]
base = "light"
primaryColor = "{c['primary']}"
backgroundColor = "{c['background']}"
secondaryBackgroundColor = "{c['well']}"
textColor = "{c['foreground']}"
font = "{b['base_font']}:{font_url(b['base_font'])}"
headingFont = "{b['heading_font']}:{font_url(b['heading_font'])}"
borderColor = "{c['line']}"

[server]
headless = true

[browser]
gatherUsageStats = false
""")
    return path


if __name__ == "__main__":
    print(f"Wrote {write_streamlit_config()}")
