"""Multipage entry point for the paper portfolio dashboard."""

from pathlib import Path
import runpy


dashboard_path = (
    Path(__file__).resolve().parents[1]
    / "paper_portfolio_dashboard.py"
)

runpy.run_path(
    str(dashboard_path),
    run_name="__main__",
)
