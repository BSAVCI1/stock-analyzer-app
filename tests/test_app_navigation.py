from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


STOCK_APP = Path(
    "stock_analysis_app.py"
)

PORTFOLIO_APP = Path(
    "paper_portfolio_dashboard.py"
)

PORTFOLIO_PAGE = Path(
    "pages/2_Paper_Portfolio.py"
)

GUIDE_PAGE = Path(
    "pages/3_App_Guide.py"
)


def test_stock_app_links_to_portfolio_and_guide() -> None:
    source = STOCK_APP.read_text(
        encoding="utf-8"
    )

    assert (
        'st.switch_page(\n'
        '            "pages/2_Paper_Portfolio.py"'
        in source
    )

    assert (
        'st.switch_page(\n'
        '            "pages/3_App_Guide.py"'
        in source
    )


def test_portfolio_app_links_to_stock_and_guide() -> None:
    source = PORTFOLIO_APP.read_text(
        encoding="utf-8"
    )

    assert (
        '"stock_analysis_app.py"'
        in source
    )

    assert (
        '"pages/3_App_Guide.py"'
        in source
    )


def test_portfolio_page_executes_dashboard() -> None:
    source = PORTFOLIO_PAGE.read_text(
        encoding="utf-8"
    )

    assert "runpy.run_path" in source

    assert (
        "paper_portfolio_dashboard.py"
        in source
    )


def test_app_guide_starts() -> None:
    app = AppTest.from_file(
        str(GUIDE_PAGE)
    )

    app.run(timeout=15)

    assert not app.exception

    assert len(app.tabs) == 2

    assert {
        tab.label
        for tab in app.tabs
    } == {
        "📈 Stock Analyzer Guide",
        "📊 Paper Portfolio Guide",
    }


def test_app_guide_documents_key_sections() -> None:
    source = GUIDE_PAGE.read_text(
        encoding="utf-8"
    )

    required_topics = (
        "Market & Trading Overview",
        "Fundamental Breakdown vs Peers",
        "Trading Expert recommendation",
        "Positions & Orders",
        "Trades & Evidence",
        "Equity & Performance",
        "Scans & Strategy",
        "Reliability",
        "Provenance",
    )

    for topic in required_topics:
        assert topic in source
