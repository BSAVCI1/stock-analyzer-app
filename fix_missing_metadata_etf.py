
from pathlib import Path
from textwrap import indent

path = Path("stock_analysis_app.py")
text = path.read_text(encoding="utf-8")

if 'is_etf = quote_type == "ETF"' in text:
    raise SystemExit("Patch already appears to be applied.")

def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise SystemExit(f"Could not find {label}.")
    return source.replace(old, new, 1)

def replace_between(
    source: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
) -> str:
    try:
        start = source.index(start_marker)
        end = source.index(end_marker, start)
    except ValueError as exc:
        raise SystemExit(
            f"Could not find patch markers: {start_marker!r} -> {end_marker!r}"
        ) from exc
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]

text = replace_once(
    text,
    "info = dict(snapshot.metadata)\n",
    """info = dict(snapshot.metadata)

quote_type = str(
    info.get("quoteType")
    or info.get("instrumentType")
    or "UNKNOWN"
).upper()
is_etf = quote_type == "ETF"
""",
    "validated metadata assignment",
)

text = replace_once(
    text,
    "    peer_list = industry_map.get(industry) or sector_map.get(sector) or popular\n",
    """    peer_list = industry_map.get(industry) or sector_map.get(sector) or popular

    if is_etf:
        etf_peer_map = {
            "VWCE.DE": ["VWRL.AS", "IUSQ.DE", "EUNL.DE"],
            "SXR8.DE": ["CSPX.L", "VUAA.DE", "IUSA.L"],
        }
        peer_list = etf_peer_map.get(ticker, [])
""",
    "automatic peer fallback",
)

text = replace_once(
    text,
    """st.caption(
    f"Currency: **{instrument_currency}**"
""",
    """st.caption(
    f"Type: **{quote_type}**"
    f" · Currency: **{instrument_currency}**"
""",
    "instrument caption",
)

helper_block = """CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CHF": "CHF ",
    "CAD": "C$",
    "AUD": "A$",
    "CNY": "CN¥",
    "HKD": "HK$",
    "SEK": "SEK ",
    "NOK": "NOK ",
    "DKK": "DKK ",
}


def numeric_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def format_money(value, currency: str, decimals: int = 0) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"

    code = str(currency or "N/A").upper()
    symbol = CURRENCY_SYMBOLS.get(code)
    sign = "-" if number < 0 else ""
    absolute_value = abs(number)

    if symbol:
        return f"{sign}{symbol}{absolute_value:,.{decimals}f}"

    return f"{sign}{absolute_value:,.{decimals}f} {code}"


def format_ratio(value, decimals: int = 2) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"
    return f"{number:.{decimals}f}"


def format_count(value) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"
    return f"{number:,.0f}"


def dividend_yield_percent(metadata: dict) -> float:
    direct = numeric_or_nan(metadata.get("dividendYield"))
    trailing = numeric_or_nan(
        metadata.get("trailingAnnualDividendYield")
    )

    if not np.isnan(direct) and not np.isnan(trailing):
        trailing_percent = trailing * 100
        direct_as_percent = direct
        direct_as_fraction = direct * 100
        return (
            direct_as_percent
            if abs(direct_as_percent - trailing_percent)
            <= abs(direct_as_fraction - trailing_percent)
            else direct_as_fraction
        )

    if not np.isnan(direct):
        return direct * 100 if abs(direct) < 0.20 else direct

    if not np.isnan(trailing):
        return trailing * 100

    return np.nan


def comparison_arrow(left, right) -> str:
    lhs = numeric_or_nan(left)
    rhs = numeric_or_nan(right)

    if np.isnan(lhs) or np.isnan(rhs) or lhs == rhs:
        return ""
    if lhs > rhs:
        return '<span class="arrow-up">▲</span>'
    return '<span class="arrow-down">▼</span>'


vol = numeric_or_nan(info.get("volume"))
avg_vol = numeric_or_nan(info.get("averageVolume"))
mc = numeric_or_nan(info.get("marketCap"))
rev = numeric_or_nan(info.get("totalRevenue"))
beta = numeric_or_nan(info.get("beta"))
dy = dividend_yield_percent(info)
"""

text = replace_between(
    text,
    'vol = info.get("volume") or 0',
    "c1, c2, c3 = st.columns(3)",
    helper_block,
)

overview_cards = """c1, c2, c3 = st.columns(3)

volume_arrow = comparison_arrow(vol, avg_vol)
average_volume_arrow = comparison_arrow(avg_vol, vol)

shares_outstanding = numeric_or_nan(info.get("sharesOutstanding"))
previous_market_cap = (
    prev_close * shares_outstanding
    if not np.isnan(shares_outstanding)
    else np.nan
)
market_cap_arrow = (
    ""
    if is_etf
    else comparison_arrow(mc, previous_market_cap)
)

peer_dividend_yields = []
peer_dividend_warnings = []

for peer_symbol in dict.fromkeys(peer_list):
    try:
        peer_metadata = yf.Ticker(peer_symbol).get_info() or {}
        peer_yield = dividend_yield_percent(peer_metadata)
        if not np.isnan(peer_yield):
            peer_dividend_yields.append(peer_yield)
    except Exception as exc:
        peer_dividend_warnings.append(
            f"{peer_symbol}: {type(exc).__name__}"
        )

if peer_dividend_warnings:
    st.warning(
        "Some peer dividend data could not be loaded: "
        + ", ".join(peer_dividend_warnings)
    )

average_peer_dividend_yield = (
    float(np.mean(peer_dividend_yields))
    if peer_dividend_yields
    else np.nan
)

dividend_arrow = comparison_arrow(
    dy,
    average_peer_dividend_yield,
)
beta_arrow = comparison_arrow(beta, 1)

market_cap_text = (
    "N/A"
    if is_etf
    else format_money(mc, instrument_currency, 0)
)
revenue_text = (
    "Not applicable to ETF"
    if is_etf
    else format_money(rev, financial_currency, 0)
)
dividend_text = "N/A" if np.isnan(dy) else f"{dy:.2f}%"
beta_text = format_ratio(beta)

c1.markdown(
    f"**Volume:** {format_count(vol)} {volume_arrow} "
    "<abbr title='Shares traded during the latest session.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c2.markdown(
    f"**Avg Volume:** {format_count(avg_vol)} {average_volume_arrow} "
    "<abbr title='Average recent trading volume.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c3.markdown(
    f"**Market Cap:** {market_cap_text} {market_cap_arrow} "
    "<abbr title='Corporate market value; unavailable for ETFs.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c4, c5, c6 = st.columns(3)
c4.markdown(
    f"**Revenue (TTM):** {revenue_text} "
    "<abbr title='Corporate trailing-twelve-month revenue.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c5.markdown(
    f"**Dividend Yield:** {dividend_text} {dividend_arrow} "
    "<abbr title='Annual dividend or distribution yield.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c6.markdown(
    f"**Beta:** {beta_text} {beta_arrow} "
    "<abbr title='Historical volatility relative to the wider market.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

overview_notes = []

if not np.isnan(vol) and not np.isnan(avg_vol):
    overview_notes.append(
        "Volume was "
        f"{'above' if vol > avg_vol else 'below'} its recent average; "
        f"{'stronger interest' if vol > avg_vol else 'muted trading'}."
    )
else:
    overview_notes.append("Volume comparison is unavailable.")

if is_etf:
    overview_notes.append(
        "Corporate market-cap, revenue and profitability metrics "
        "are not applicable to this ETF."
    )
else:
    if not np.isnan(mc):
        size_label = "small" if mc < 1e9 else "mid/large"
        overview_notes.append(
            f"Market cap {format_money(mc, instrument_currency, 0)} "
            f"({size_label}-cap)."
        )
    if not np.isnan(rev):
        overview_notes.append(
            f"TTM revenue {format_money(rev, financial_currency, 0)}."
        )

if not np.isnan(dy):
    overview_notes.append(
        f"Dividend yield {dy:.2f}% "
        f"({'pays a distribution' if dy > 0 else 'no payout reported'})."
    )
else:
    overview_notes.append("Dividend yield is unavailable.")

if not np.isnan(beta):
    overview_notes.append(
        f"Beta {beta:.2f} "
        f"({'high' if beta > 1 else 'lower'} relative volatility)."
    )
else:
    overview_notes.append("Beta is unavailable.")

st.markdown(
    "<div class='card-dark'>🔍 "
    + " ".join(overview_notes)
    + "</div>",
    unsafe_allow_html=True,
)
"""

text = replace_between(
    text,
    "c1, c2, c3 = st.columns(3)",
    "# --- EXTENDED FUNDAMENTALS vs PEERS ---",
    overview_cards,
)

fundamentals_marker = "# --- EXTENDED FUNDAMENTALS vs PEERS ---"
technical_marker = "# --- TECHNICAL PARAMETER CONTROLS ---"

start = text.index(fundamentals_marker)
end = text.index(technical_marker, start)
existing_fundamentals = text[
    start + len(fundamentals_marker):end
].lstrip("\n")

wrapped_fundamentals = (
    fundamentals_marker
    + """
if is_etf:
    st.markdown(
        "<div class='card'><h2>📑 Fund Structure</h2></div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Corporate valuation, profitability, leverage and quarterly "
        "earnings metrics are not applicable to ETFs. "
        "ETF-specific holdings, fees, assets and tracking metrics "
        "will be added in a later implementation step."
    )
else:
"""
    + indent(existing_fundamentals.rstrip(), "    ")
)

text = (
    text[:start]
    + wrapped_fundamentals.rstrip()
    + "\n\n"
    + text[end:]
)

path.write_text(text, encoding="utf-8")
print("Missing metadata and ETF presentation patch applied.")
