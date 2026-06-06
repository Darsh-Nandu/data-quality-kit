from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

class CheckSeveruty(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

class CheckIssue(BaseModel):
    """ Single issue found during a check."""
    cloumn: str | None = None
    message: str
    severity: CheckSeveruty = CheckSeveruty.WARN
    extra: dict[str, Any] = Field(default_factory=list)

class CheckResult(BaseModel):
    """
    Output of a quality check.
    """
    check_name: str
    score: float = Field(ge=0.0, le=1.0, description="Normalized 0-1 quality score foe this check")
    severity: CheckSeveruty = CheckSeveruty.PASS
    issues: list[CheckIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict, description="Raw numbers produced by the check")
    description: str = ""

    @property
    def passed(self) -> bool:
        return self.severity == CheckSeveruty.PASS

    @property
    def n_issues(self) -> int:
        return len(self.issues)

    def add_issue(
        self,
        message: str,
        column: str | None = None,
        severity: CheckSeveruty = CheckSeveruty.WARN,
        **extra: Any,
    ) -> None:
        self.issues.append(CheckIssue(column=column, message=message, severity=severity, extra=extra))

        if severity == CheckSeveruty.FAIL:
            self.severity = CheckSeveruty.FAIL
        elif severity == CheckSeveruty.WARN and self.severity == CheckSeveruty.PASS:
            self.severity = CheckSeveruty.WARN

class BaseCheck:
    """
    Abstract base class for DQK checks.
    """
    name: str  = "base_check"
    description: str = ""
    weight: float = 1.0 # used in weighted scoring

    def run(self, dataset: Any) -> CheckResult:
        raise NotImplementedError

    def _empty_result(self) -> CheckResult:
        return CheckResult(
            check_name=self.name,
            score=1.0,
            severity=CheckSeveruty.PASS,
            description=self.description,
        )
