"""DataQualityKit CLI — powered by Typer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="dqk",
    help="DataQualityKit — dataset quality testing for ML teams.",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    source: str = typer.Argument(..., help="Dataset path, HuggingFace ID, or SQL connection string."),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Force format: csv, json, parquet, hf, sql."),
    split: str = typer.Option("train", "--split", "-s", help="HuggingFace split (default: train)."),
    checks: Optional[str] = typer.Option(None, "--checks", "-c", help="Comma-separated check names to run."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file (.html or .json)."),
    fail_under: float = typer.Option(0.0, "--fail-under", help="Exit with code 1 if score < this value."),
) -> None:
    """Run quality checks on a dataset and print a summary."""
    from dqk.core.dataset import DQKDataset

    check_list = [c.strip() for c in checks.split(",")] if checks else None

    with console.status(f"Loading dataset from [bold]{source}[/bold]..."):
        try:
            ds = DQKDataset.from_csv(source) if (format or "").lower() == "csv" \
                else _auto_load(source, format, split)
        except Exception as e:
            console.print(f"[red]Error loading dataset:[/red] {e}")
            raise typer.Exit(1) from e

    console.print(
        f"[dim]Loaded:[/dim] {ds.shape[0]:,} rows × {ds.shape[1]} cols "
        f"([dim]{ds.schema.format}[/dim])"
    )

    with console.status("Running checks..."):
        report = ds.run_checks(checks=check_list)

    # Print score
    grade_color = {"A": "green", "B": "yellow", "C": "yellow", "D": "red", "F": "red"}
    color = grade_color.get(report.score.grade, "white")
    console.print(
        f"\n[bold]Quality Score:[/bold] "
        f"[{color} bold]{report.score.overall:.1f}/100  ({report.score.grade})[/{color} bold]"
        f"  [{report.score.severity.value}]{report.score.severity.value.upper()}[/{report.score.severity.value}]"
    )

    # Print per-check table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Severity", justify="center")
    table.add_column("Issues", justify="right")

    for r in report.results:
        sev_color = {"pass": "green", "warn": "yellow", "fail": "red", "skip": "dim"}.get(
            r.severity.value, "white"
        )
        table.add_row(
            r.check_name,
            f"{r.score:.3f}",
            f"[{sev_color}]{r.severity.value}[/{sev_color}]",
            str(r.n_issues),
        )
    console.print(table)

    # Print issues
    if report.n_issues > 0:
        console.print(f"\n[bold]Issues ({report.n_issues} total):[/bold]")
        for r in report.results:
            for issue in r.issues:
                icon = {"fail": "✗", "warn": "⚠", "pass": "✓", "skip": "—"}.get(
                    issue.severity.value, "·"
                )
                col = {"fail": "red", "warn": "yellow"}.get(issue.severity.value, "dim")
                col_label = f" [dim]({issue.column})[/dim]" if issue.column else ""
                console.print(f"  [{col}]{icon}[/{col}]{col_label} {issue.message}")

    # Save output
    if output:
        report.save(output)
        console.print(f"\n[dim]Report saved to:[/dim] [bold]{output}[/bold]")

    # Fail-under exit code
    if report.score.overall < fail_under:
        console.print(
            f"\n[red]Score {report.score.overall:.1f} < fail-under threshold {fail_under}[/red]"
        )
        raise typer.Exit(1)


@app.command()
def schema(
    source: str = typer.Argument(..., help="Dataset path or HuggingFace ID."),
    format: Optional[str] = typer.Option(None, "--format", "-f"),
    split: str = typer.Option("train", "--split", "-s"),
) -> None:
    """Print the inferred schema of a dataset."""
    from dqk.core.dataset import DQKDataset

    ds = _auto_load(source, format, split)
    table = Table(show_header=True, header_style="bold", title=f"Schema: {source}")
    table.add_column("Column")
    table.add_column("Dtype")
    table.add_column("Role")
    table.add_column("Missing%", justify="right")
    table.add_column("N Unique", justify="right")

    for col in ds.schema.columns:
        missing = f"{col.missing_rate:.1%}" if col.missing_rate is not None else "—"
        table.add_row(
            col.name,
            col.dtype.value,
            col.role.value,
            missing,
            str(col.n_unique) if col.n_unique is not None else "—",
        )
    console.print(table)


def _auto_load(
    source: str,
    format: Optional[str],
    split: str,
) -> "DQKDataset":  # type: ignore[name-defined]
    from dqk.core.dataset import DQKDataset
    from dqk.core.loader import load
    from dqk.core.loader import infer_schema

    df, src, fmt = load(source, format=format, split=split)
    from dqk.core.loader import infer_schema
    return DQKDataset(df, infer_schema(df, source=src, fmt=fmt))


if __name__ == "__main__":
    app()
