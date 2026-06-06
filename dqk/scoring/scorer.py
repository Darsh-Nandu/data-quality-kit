"""
Scoring engine and QualityReport for DataQualityKit.

Aggregates CheckResult objects into a single weighted quality score
and provides save/display helpers.
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

    # Summary helpers
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

    # Output / persistence

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def save(self, path: str | Path) -> None:
        """
        Save the report. Format is inferred from the file extension:
        ``.html`` → HTML report, ``.json`` → JSON.
        """
        path = Path(path)
        if path.suffix == ".json":
            self.save_json(path)
        else:
            self._save_html(path)

    def _save_html(self, path: Path) -> None:
        """Generate a simple standalone HTML report (full Plotly dashboard in Phase 5)."""
        from jinja2 import Environment, BaseLoader

        template_str = _HTML_TEMPLATE
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(template_str)
        html = tmpl.render(report=self)
        path.write_text(html, encoding="utf-8")

    # Jupyter display
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


# Check runner
# Default check registry (extended in later phases)
_DEFAULT_CHECKS = ["completeness", "validity", "uniqueness"]


def _build_checks(checks: list[str] | None) -> list[Any]:
    """Instantiate check objects from name strings."""
    from dqk.checks.completeness import CompletenessCheck
    from dqk.checks.validity import ValidityCheck
    from dqk.checks.uniqueness import UniquenessCheck

    registry = {
        "completeness": CompletenessCheck,
        "validity": ValidityCheck,
        "uniqueness": UniquenessCheck,
    }
    names = checks or _DEFAULT_CHECKS
    result = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown check: '{name}'. Available: {list(registry)}")
        result.append(registry[name]())
    return result


def _aggregate_score(results: list[CheckResult]) -> QualityScore:
    """Weighted average of check scores → 0-100 overall score."""
    from dqk.checks.completeness import CompletenessCheck
    from dqk.checks.validity import ValidityCheck
    from dqk.checks.uniqueness import UniquenessCheck

    weight_map = {
        "completeness": CompletenessCheck.weight,
        "validity": ValidityCheck.weight,
        "uniqueness": UniquenessCheck.weight,
    }

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
    check_objects = _build_checks(checks)
    results: list[CheckResult] = []
    for chk in check_objects:
        try:
            res = chk.run(dataset)
        except Exception as exc:  # noqa: BLE001
            from dqk.checks.base import CheckResult as CR
            res = CR(
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

# Minimal HTML template (Plotly dashboard added in Phase 5)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DataQualityKit Report — {{ report.dataset_source }}</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; }
    h1 { font-size: 24px; font-weight: 600; }
    .score { font-size: 48px; font-weight: 700; }
    .grade-A { color: #22c55e; } .grade-B { color: #84cc16; } .grade-C { color: #f59e0b; }
    .grade-D { color: #f97316; } .grade-F { color: #ef4444; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e5e5; }
    th { background: #f9f9f9; font-weight: 600; }
    .pass { color: #22c55e; } .warn { color: #f59e0b; } .fail { color: #ef4444; }
    .issue { font-size: 13px; color: #555; margin: 2px 0 2px 8px; }
  </style>
</head>
<body>
  <h1>DataQualityKit Report</h1>
  <p><b>Source:</b> {{ report.dataset_source }} &nbsp;|&nbsp;
     <b>Rows:</b> {{ report.n_rows }} &nbsp;|&nbsp;
     <b>Cols:</b> {{ report.n_cols }}</p>

  <div class="score grade-{{ report.score.grade }}">
    {{ "%.1f"|format(report.score.overall) }}/100 &nbsp; ({{ report.score.grade }})
  </div>

  <table>
    <thead><tr><th>Check</th><th>Score</th><th>Severity</th><th>Issues</th></tr></thead>
    <tbody>
    {% for r in report.results %}
      <tr>
        <td>{{ r.check_name }}</td>
        <td>{{ "%.3f"|format(r.score) }}</td>
        <td class="{{ r.severity.value }}">{{ r.severity.value }}</td>
        <td>{{ r.n_issues }}</td>
      </tr>
      {% for issue in r.issues %}
        <tr>
          <td colspan="4" class="issue">
            ⚠ {{ issue.message }}
          </td>
        </tr>
      {% endfor %}
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
