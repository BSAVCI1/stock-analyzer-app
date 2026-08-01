from __future__ import annotations

import ast
from pathlib import Path
import re

from streamlit.testing.v1 import AppTest

from src.paper import PaperRepository


APP_PATH = Path(
    "paper_portfolio_dashboard.py"
)

PACKAGE_PATH = Path(
    "src/portfolio_dashboard"
)


def test_dashboard_starts_without_configuration(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "PAPER_ACCOUNT_ID",
        raising=False,
    )

    monkeypatch.delenv(
        "PAPER_DATABASE_PATH",
        raising=False,
    )

    app = AppTest.from_file(
        str(APP_PATH)
    )

    app.run(timeout=15)

    assert not app.exception

    assert any(
        "Paper Portfolio"
        in title.value
        for title in app.title
    )

    assert any(
        "Enter an existing paper account ID"
        in item.value
        for item in app.info
    )


def test_dashboard_renders_existing_account(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = (
        tmp_path / "dashboard-app.db"
    )

    repository = PaperRepository(
        database_path
    )

    account = repository.create_account(
        name="Dashboard App Test",
        base_currency="USD",
        starting_balance="10000",
    )

    monkeypatch.setenv(
        "PAPER_DATABASE_PATH",
        str(database_path),
    )

    monkeypatch.setenv(
        "PAPER_ACCOUNT_ID",
        account.account_id,
    )

    app = AppTest.from_file(
        str(APP_PATH)
    )

    app.run(timeout=15)

    assert not app.exception

    assert len(app.tabs) == 7

    assert any(
        account.account_id
        in caption.value
        for caption in app.caption
    )

    tab_labels = {
        tab.label
        for tab in app.tabs
    }

    assert tab_labels == {
        "Overview",
        "Positions & Orders",
        "Trades & Evidence",
        "Equity & Performance",
        "Scans & Strategy",
        "Reliability",
        "Provenance",
    }


def test_dashboard_has_no_network_provider_imports() -> None:
    tree = ast.parse(
        APP_PATH.read_text(
            encoding="utf-8"
        )
    )

    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name.split(".")[0]
                for alias in node.names
            )

        if isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imported_modules.add(
                    node.module.split(".")[0]
                )

    assert imported_modules.isdisjoint(
        {
            "yfinance",
            "requests",
            "feedparser",
            "urllib",
            "httpx",
        }
    )


def test_dashboard_app_uses_only_read_model_domain() -> None:
    tree = ast.parse(
        APP_PATH.read_text(
            encoding="utf-8"
        )
    )

    forbidden_direct_domains = {
        "src.paper",
        "src.scanner",
        "src.automation",
        "src.jobs",
        "src.notifications",
        "src.data",
    }

    imported_from = {
        node.module
        for node in ast.walk(tree)
        if (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and node.module
        )
    }

    assert imported_from.isdisjoint(
        forbidden_direct_domains
    )

    assert (
        "src.portfolio_dashboard"
        in imported_from
    )


def test_dashboard_package_is_read_only() -> None:
    source = "\n".join(
        path.read_text(
            encoding="utf-8"
        )
        for path in sorted(
            PACKAGE_PATH.glob("*.py")
        )
    )

    mutation_pattern = re.compile(
        r"""
        \bINSERT\s+INTO\b
        |
        \bUPDATE\s+\w+\s+SET\b
        |
        \bDELETE\s+FROM\b
        |
        \bplace_order\s*\(
        |
        \bsubmit_order\s*\(
        |
        \bsend_order\s*\(
        """,
        re.IGNORECASE
        | re.VERBOSE,
    )

    assert (
        mutation_pattern.search(source)
        is None
    )


def test_dashboard_guide_documents_traceability() -> None:
    guide = Path(
        "docs/PAPER_PORTFOLIO_DASHBOARD.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "streamlit run "
        "paper_portfolio_dashboard.py"
        in guide
    )

    assert "source SQLite tables" in guide
    assert "persisted record identifiers" in guide
    assert "deterministic calculations" in guide
    assert "read-only" in guide
