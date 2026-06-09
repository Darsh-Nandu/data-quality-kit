"""
Scoring engine and QualityReport for DataQualityKit.

Aggregates CheckResult objects into a single weighted quality score
and provides a rich interactive Plotly HTML report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from dqk.checks.base import CheckResult, CheckSeverity

if TYPE_CHECKING:
    from dqk.core.dataset import DQKDataset


# Score model
class QualityScore(BaseModel):
    """Aggregated quality score (0–100)."""

    overall: float = Field(ge=0.0, le=100.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    severity: CheckSeverity = CheckSeverity.PASS

    @property
    def grade(self) -> str:
        if self.overall >= 90:
            return "A"
        if self.overall >= 75:
            return "B"
        if self.overall >= 60:
            return "C"
        if self.overall >= 40:
            return "D"
        return "F"


# Report model
class QualityReport(BaseModel):
    """Full quality report returned by DQKDataset.run_checks()."""

    dataset_source: str
    n_rows: int
    n_cols: int
    score: QualityScore
    results: list[CheckResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def passed(self) -> bool:
        return self.score.severity != CheckSeverity.FAIL

    @property
    def n_issues(self) -> int:
        return sum(r.n_issues for r in self.results)

    def failed_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == CheckSeverity.FAIL]

    def warned_checks(self) -> list[CheckResult]:
        return [r for r in self.results if r.severity == CheckSeverity.WARN]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def save(self, path: str | Path) -> None:
        """
        Save the report. Format is inferred from the file extension:
        ``.html`` → interactive Plotly dashboard, ``.json`` → JSON.
        """
        path = Path(path)
        if path.suffix == ".json":
            self.save_json(path)
        else:
            self._save_html(path)

    def _save_html(self, path: Path) -> None:
        html = _build_plotly_html(self)
        path.write_text(html, encoding="utf-8")

    def _repr_html_(self) -> str:
        grade_color = {
            "A": "#22c55e", "B": "#84cc16", "C": "#f59e0b",
            "D": "#f97316", "F": "#ef4444",
        }
        color = grade_color.get(self.score.grade, "#888")
        rows = "".join(
            f"<tr>"
            f"<td>{r.check_name}</td>"
            f"<td>{r.score:.2f}</td>"
            f"<td>{r.severity.value}</td>"
            f"<td>{r.n_issues}</td>"
            f"</tr>"
            for r in self.results
        )
        return f"""
        <div style="font-family: monospace; font-size: 13px;">
          <b>QualityReport</b> — {self.dataset_source}
          &nbsp;|&nbsp; {self.n_rows:,} rows × {self.n_cols} cols
          &nbsp;|&nbsp; Score: <span style="color:{color}; font-size:18px; font-weight:bold;">
            {self.score.overall:.1f}/100 ({self.score.grade})</span>
          <br/><br/>
          <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse">
            <thead>
              <tr style="background:#f5f5f5">
                <th>check</th><th>score</th><th>severity</th><th># issues</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """


# Plotly HTML dashboard

def _build_plotly_html(report: QualityReport) -> str:
    """Build a standalone interactive HTML dashboard using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        # Fallback to plain HTML if plotly not installed
        return _build_fallback_html(report)

    grade_colors = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}
    sev_colors = {"pass": "#22c55e", "warn": "#f59e0b", "fail": "#ef4444", "skip": "#94a3b8"}
    grade_color = grade_colors.get(report.score.grade, "#888")

    # Figure 1: Radial gauge for overall score
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=report.score.overall,
        number={"suffix": "/100", "font": {"size": 36, "color": grade_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": grade_color, "thickness": 0.3},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 60], "color": "#fef3c7"},
                {"range": [60, 75], "color": "#fef9c3"},
                {"range": [75, 90], "color": "#dcfce7"},
                {"range": [90, 100], "color": "#bbf7d0"},
            ],
            "threshold": {
                "line": {"color": grade_color, "width": 4},
                "thickness": 0.8,
                "value": report.score.overall,
            },
        },
        title={"text": f"Overall Quality Grade: <b>{report.score.grade}</b>", "font": {"size": 18}},
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=40, r=40))

    # Figure 2: Per-check bar chart
    active_results = [r for r in report.results if r.severity != CheckSeverity.SKIP]
    check_names = [r.check_name.replace("_", " ").title() for r in active_results]
    check_scores = [r.score * 100 for r in active_results]
    bar_colors = [sev_colors.get(r.severity.value, "#888") for r in active_results]

    fig_bars = go.Figure(go.Bar(
        x=check_scores,
        y=check_names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}" for s in check_scores],
        textposition="outside",
    ))
    fig_bars.update_layout(
        title="Check Scores (0–100)",
        xaxis=dict(range=[0, 110], title="Score"),
        yaxis=dict(autorange="reversed"),
        height=max(200, len(active_results) * 50 + 80),
        margin=dict(t=50, b=40, l=160, r=60),
    )
    fig_bars.add_vline(x=75, line_dash="dash", line_color="#6b7280", annotation_text="B threshold")

    # Figure 3: Issue breakdown donut
    fail_count = sum(1 for r in report.results for i in r.issues if i.severity == CheckSeverity.FAIL)
    warn_count = sum(1 for r in report.results for i in r.issues if i.severity == CheckSeverity.WARN)
    pass_count = max(0, len(report.results) - len(report.failed_checks()) - len(report.warned_checks()))

    if report.n_issues > 0:
        fig_donut = go.Figure(go.Pie(
            labels=["FAIL", "WARN", "PASS"],
            values=[fail_count, warn_count, pass_count],
            hole=0.55,
            marker_colors=["#ef4444", "#f59e0b", "#22c55e"],
        ))
        fig_donut.update_layout(
            title="Issue Breakdown",
            height=280,
            margin=dict(t=50, b=0, l=20, r=20),
        )
        donut_html = pio.to_html(fig_donut, full_html=False, include_plotlyjs=False)
    else:
        donut_html = "<div style='text-align:center;padding:60px;color:#22c55e;font-size:18px'>✓ No issues found</div>"

    # Serialize figures
    gauge_html = pio.to_html(fig_gauge, full_html=False, include_plotlyjs="cdn")
    bars_html = pio.to_html(fig_bars, full_html=False, include_plotlyjs=False)

    # Issue table HTML
    issue_rows = ""
    for r in report.results:
        for issue in r.issues:
            sev = issue.severity.value
            badge_color = {"fail": "#ef4444", "warn": "#f59e0b", "pass": "#22c55e", "skip": "#94a3b8"}.get(sev, "#888")
            col_label = f"<code>{issue.column}</code>" if issue.column else "—"
            issue_rows += f"""
            <tr>
              <td>{r.check_name}</td>
              <td>{col_label}</td>
              <td><span style="background:{badge_color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{sev.upper()}</span></td>
              <td style="font-size:13px;color:#374151">{issue.message}</td>
            </tr>"""

    issue_table = f"""
    <table style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif">
      <thead>
        <tr style="background:#f3f4f6">
          <th style="padding:10px;text-align:left;border-bottom:2px solid #e5e7eb">Check</th>
          <th style="padding:10px;text-align:left;border-bottom:2px solid #e5e7eb">Column</th>
          <th style="padding:10px;text-align:left;border-bottom:2px solid #e5e7eb">Severity</th>
          <th style="padding:10px;text-align:left;border-bottom:2px solid #e5e7eb">Message</th>
        </tr>
      </thead>
      <tbody>{issue_rows}</tbody>
    </table>
    """ if issue_rows else "<p style='color:#22c55e;font-size:15px'>✓ No issues detected.</p>"

    # Assemble full page
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DQK Report — {report.dataset_source}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; }}
    .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
               color: #fff; padding: 32px 48px; }}
    .header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
    .header .meta {{ font-size: 14px; opacity: 0.8; margin-top: 8px; }}
    .content {{ max-width: 1200px; margin: 32px auto; padding: 0 24px; }}
    .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.1);
              padding: 24px; margin-bottom: 24px; }}
    .card h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; color: #1e293b; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }}
    .stat {{ text-align: center; padding: 16px; background: #f8fafc;
              border-radius: 8px; border: 1px solid #e2e8f0; }}
    .stat-value {{ font-size: 32px; font-weight: 700; color: {grade_color}; }}
    .stat-label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
    table td, table th {{ padding: 10px 12px; border-bottom: 1px solid #f1f5f9; }}
    table tr:hover {{ background: #f8fafc; }}
    .footer {{ text-align: center; color: #94a3b8; font-size: 12px; padding: 32px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>DataQualityKit Report</h1>
    <div class="meta">
      Source: <strong>{report.dataset_source}</strong> &nbsp;·&nbsp;
      {report.n_rows:,} rows × {report.n_cols} columns &nbsp;·&nbsp;
      {len(report.results)} checks &nbsp;·&nbsp;
      {report.n_issues} issue{"s" if report.n_issues != 1 else ""}
    </div>
  </div>

  <div class="content">

    <!-- KPI strip -->
    <div class="grid-3" style="margin-bottom:24px">
      <div class="stat">
        <div class="stat-value">{report.score.overall:.1f}</div>
        <div class="stat-label">Overall Score / 100</div>
      </div>
      <div class="stat">
        <div class="stat-value" style="color:{'#ef4444' if report.failed_checks() else '#22c55e'}">{len(report.failed_checks())}</div>
        <div class="stat-label">Failed Checks</div>
      </div>
      <div class="stat">
        <div class="stat-value" style="color:{'#f59e0b' if report.warned_checks() else '#22c55e'}">{len(report.warned_checks())}</div>
        <div class="stat-label">Warned Checks</div>
      </div>
    </div>

    <!-- Charts row -->
    <div class="grid-2">
      <div class="card">
        {gauge_html}
      </div>
      <div class="card">
        {donut_html}
      </div>
    </div>

    <!-- Bar chart -->
    <div class="card">
      {bars_html}
    </div>

    <!-- Issues table -->
    <div class="card">
      <h2>Issues ({report.n_issues})</h2>
      {issue_table}
    </div>

  </div>
  <div class="footer">Generated by <strong>DataQualityKit</strong></div>
</body>
</html>"""


def _build_fallback_html(report: QualityReport) -> str:
    """Plain HTML fallback when Plotly is not available."""
    from jinja2 import BaseLoader, Environment
    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_FALLBACK_TEMPLATE)
    return tmpl.render(report=report)


# Check registry

_CHECK_REGISTRY: dict[str, type] = {}


def register_check(cls: type) -> type:
    """Decorator to register a custom check class."""
    _CHECK_REGISTRY[cls.name] = cls
    return cls


def _build_default_registry() -> dict[str, type]:
    from dqk.checks.completeness import CompletenessCheck
    from dqk.checks.validity import ValidityCheck
    from dqk.checks.uniqueness import UniquenessCheck
    from dqk.checks.distribution import DistributionCheck
    from dqk.checks.text_quality import TextQualityCheck
    from dqk.checks.label_quality import LabelQualityCheck

    built_in = {
        "completeness": CompletenessCheck,
        "validity": ValidityCheck,
        "uniqueness": UniquenessCheck,
        "distribution": DistributionCheck,
        "text_quality": TextQualityCheck,
        "label_quality": LabelQualityCheck,
    }
    return {**built_in, **_CHECK_REGISTRY}


_DEFAULT_CHECKS = ["completeness", "validity", "uniqueness", "distribution", "text_quality", "label_quality"]


def available_checks() -> list[str]:
    """Return a list of all registered check names."""
    return list(_build_default_registry().keys())


def _build_checks(checks: list[str] | None) -> list[Any]:
    registry = _build_default_registry()
    names = checks or _DEFAULT_CHECKS
    result = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown check: '{name}'. Available: {list(registry)}")
        result.append(registry[name]())
    return result


def _aggregate_score(results: list[CheckResult]) -> QualityScore:
    registry = _build_default_registry()

    # Build weight map dynamically from registered checks
    weight_map: dict[str, float] = {}
    for name, cls in registry.items():
        weight_map[name] = getattr(cls, "weight", 1.0)

    total_weight = 0.0
    weighted_sum = 0.0
    breakdown: dict[str, float] = {}
    worst_severity = CheckSeverity.PASS

    severity_rank = {
        CheckSeverity.PASS: 0,
        CheckSeverity.SKIP: 0,
        CheckSeverity.WARN: 1,
        CheckSeverity.FAIL: 2,
    }

    for r in results:
        if r.severity == CheckSeverity.SKIP:
            continue  # Don't penalize skipped checks
        w = weight_map.get(r.check_name, 1.0)
        weighted_sum += r.score * w
        total_weight += w
        breakdown[r.check_name] = round(r.score * 100, 1)
        if severity_rank[r.severity] > severity_rank[worst_severity]:
            worst_severity = r.severity

    overall = (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0
    return QualityScore(
        overall=round(overall, 2),
        breakdown=breakdown,
        severity=worst_severity,
    )


def run_all_checks(
    dataset: "DQKDataset",
    checks: list[str] | None = None,
    label_col: str | None = None,
) -> QualityReport:
    """Run the requested checks and return a QualityReport."""
    if label_col:
        # Override schema role for the specified column
        for col in dataset.schema.columns:
            if col.name == label_col:
                from dqk.core.schema import ColumnRole
                col.role = ColumnRole.LABEL
                break

    check_objects = _build_checks(checks)
    results: list[CheckResult] = []
    for chk in check_objects:
        try:
            res = chk.run(dataset)
        except Exception as exc:
            res = CheckResult(
                check_name=chk.name,
                score=0.0,
                severity=CheckSeverity.FAIL,
                description=f"Check raised an exception: {exc}",
            )
        results.append(res)

    score = _aggregate_score(results)
    return QualityReport(
        dataset_source=dataset.schema.source,
        n_rows=dataset.schema.n_rows,
        n_cols=dataset.schema.n_cols,
        score=score,
        results=results,
    )


_FALLBACK_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><title>DQK Report</title>
  <style>body{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 24px;}
  .score{font-size:48px;font-weight:700;}
  .grade-A{color:#22c55e}.grade-B{color:#84cc16}.grade-C{color:#f59e0b}
  .grade-D{color:#f97316}.grade-F{color:#ef4444}
  table{width:100%;border-collapse:collapse;margin-top:16px;}
  th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #e5e5e5;}
  th{background:#f9f9f9;font-weight:600;}
  .pass{color:#22c55e}.warn{color:#f59e0b}.fail{color:#ef4444}
  .issue{font-size:13px;color:#555;margin:2px 0 2px 8px;}</style>
</head>
<body>
  <h1>DataQualityKit Report</h1>
  <p><b>Source:</b> {{ report.dataset_source }} | <b>Rows:</b> {{ report.n_rows }} | <b>Cols:</b> {{ report.n_cols }}</p>
  <div class="score grade-{{ report.score.grade }}">{{ "%.1f"|format(report.score.overall) }}/100 ({{ report.score.grade }})</div>
  <table>
    <thead><tr><th>Check</th><th>Score</th><th>Severity</th><th>Issues</th></tr></thead>
    <tbody>
    {% for r in report.results %}
      <tr><td>{{ r.check_name }}</td><td>{{ "%.3f"|format(r.score) }}</td>
      <td class="{{ r.severity.value }}">{{ r.severity.value }}</td><td>{{ r.n_issues }}</td></tr>
      {% for issue in r.issues %}
      <tr><td colspan="4" class="issue">⚠ {{ issue.message }}</td></tr>
      {% endfor %}
    {% endfor %}
    </tbody>
  </table>
</body></html>"""