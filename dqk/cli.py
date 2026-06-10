"""DataQualityKit CLI — powered by Typer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from dqk.core.dataset import DQKDataset

app = typer.Typer(
    name="dqk",
    help="DataQualityKit — dataset quality testing for ML teams.",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    source: str = typer.Argument(
        ..., help="Dataset path, HuggingFace ID, or SQL connection string."
    ),
    format: str | None = typer.Option(
        None, "--format", "-f", help="Force format: csv, json, parquet, hf, sql."
    ),
    split: str = typer.Option("train", "--split", "-s", help="HuggingFace split (default: train)."),
    checks: str | None = typer.Option(
        None, "--checks", "-c", help="Comma-separated check names to run."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save report to file (.html or .json)."
    ),
    fail_under: float = typer.Option(
        0.0, "--fail-under", help="Exit with code 1 if score < this value."
    ),
) -> None:
    """Run quality checks on a dataset and print a summary."""
    from dqk.core.dataset import DQKDataset

    check_list = [c.strip() for c in checks.split(",")] if checks else None

    with console.status(f"Loading dataset from [bold]{source}[/bold]..."):
        try:
            ds = (
                DQKDataset.from_csv(source)
                if (format or "").lower() == "csv"
                else _auto_load(source, format, split)
            )
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
        f"  [{report.score.severity.value}]"
        f"{report.score.severity.value.upper()}"
        f"[/{report.score.severity.value}]"
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
    format: str | None = typer.Option(None, "--format", "-f"),
    split: str = typer.Option("train", "--split", "-s"),
) -> None:
    """Print the inferred schema of a dataset."""

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
    format: str | None,
    split: str,
) -> DQKDataset:
    from dqk.core.dataset import DQKDataset
    from dqk.core.loader import infer_schema, load

    df, src, fmt = load(source, format=format, split=split)
    return DQKDataset(df, infer_schema(df, source=src, fmt=fmt))


if __name__ == "__main__":
    app()


@app.command()
def compare(
    reference: str = typer.Argument(..., help="Reference (baseline) dataset path."),
    current: str = typer.Argument(..., help="Current dataset to compare against the baseline."),
    ref_format: str | None = typer.Option(None, "--ref-format", help="Force format for reference."),
    cur_format: str | None = typer.Option(None, "--cur-format", help="Force format for current."),
    columns: str | None = typer.Option(
        None, "--columns", "-c", help="Comma-separated columns to compare."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Save drift report to .json file."
    ),
) -> None:
    """Compare two datasets for distribution drift (PSI, KS, chi-squared)."""
    from dqk.drift import DriftSeverity, compare_datasets

    with console.status("Loading datasets..."):
        ref_ds = _auto_load(reference, ref_format, "train")
        cur_ds = _auto_load(current, cur_format, "train")

    col_list = [c.strip() for c in columns.split(",")] if columns else None

    with console.status("Running drift analysis..."):
        drift_report = compare_datasets(ref_ds, cur_ds, columns=col_list)

    # Schema diff
    diff = drift_report.schema_diff
    if diff["added"] or diff["removed"] or diff["type_changed"]:
        console.print("\n[bold yellow]Schema Differences:[/bold yellow]")
        if diff["added"]:
            console.print(f"  [green]+ Added columns:[/green] {diff['added']}")
        if diff["removed"]:
            console.print(f"  [red]- Removed columns:[/red] {diff['removed']}")
        if diff["type_changed"]:
            for col, change in diff["type_changed"].items():
                console.print(
                    f"  [yellow]~ Type changed:[/yellow] '{col}' {change['from']} → {change['to']}"
                )

    # Drift table
    sev_color = {
        DriftSeverity.NONE: "green",
        DriftSeverity.MODERATE: "yellow",
        DriftSeverity.SEVERE: "red",
    }
    table = Table(show_header=True, header_style="bold", title="Column Drift Analysis")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Severity", justify="center")
    table.add_column("PSI / JS-div", justify="right")
    table.add_column("p-value", justify="right")

    for r in drift_report.column_results:
        color = sev_color[r.severity]
        stat = (
            f"{r.psi:.4f}"
            if r.psi is not None
            else f"{r.js_divergence:.4f}"
            if r.js_divergence is not None
            else "—"
        )
        pval = (
            f"{r.ks_pvalue:.4f}"
            if r.ks_pvalue is not None
            else f"{r.chi2_pvalue:.4f}"
            if r.chi2_pvalue is not None
            else "—"
        )
        table.add_row(
            r.column,
            r.dtype,
            f"[{color}]{r.severity.value.upper()}[/{color}]",
            stat,
            pval,
        )
    console.print(table)

    # Summary
    n_drifted = len(drift_report.drifted_columns)
    overall = drift_report.overall_severity
    color = sev_color[overall]
    console.print(
        f"\n[bold]Overall drift:[/bold] [{color}]{overall.value.upper()}[/{color}] "
        f"— {n_drifted}/{len(drift_report.column_results)} columns drifted"
    )

    if output:
        import json

        output.write_text(json.dumps(drift_report.to_dict(), indent=2, default=str))
        console.print(f"[dim]Drift report saved to:[/dim] [bold]{output}[/bold]")


@app.command(name="list-checks")
def list_checks() -> None:
    """List all available quality checks."""
    from dqk.scoring.scorer import available_checks

    table = Table(show_header=True, header_style="bold", title="Available Quality Checks")
    table.add_column("Check Name")
    table.add_column("Description")
    for name in available_checks():
        from dqk.scoring.scorer import _build_default_registry

        cls = _build_default_registry()[name]
        table.add_row(name, getattr(cls, "description", ""))
    console.print(table)
