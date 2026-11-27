"""Pytest execution and JUnit report helpers for the Software Testing Center."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent
CATEGORY_MARKERS = {
    "Unit Testing": "unit",
    "Integration Testing": "integration",
    "Functional Testing": "functional",
    "Boundary Testing": "boundary",
    "System Testing": "system",
    "Regression Testing": None,
}
_CATEGORY_MARKERS = tuple(
    marker for marker in CATEGORY_MARKERS.values() if marker is not None
)


class TestExecutionError(RuntimeError):
    """Raised when pytest cannot produce a readable execution report."""


@dataclass(frozen=True)
class TestCaseResult:
    case_id: str
    category: str
    description: str
    expected_result: str
    actual_result: str
    status: str
    duration_seconds: float
    failure_details: str


@dataclass(frozen=True)
class TestRunResult:
    selected_category: str
    exit_code: int
    duration_seconds: float
    cases: list[TestCaseResult]
    junit_xml: str
    stdout: str
    stderr: str

    @property
    def passed(self) -> int:
        return sum(case.status == "PASS" for case in self.cases)

    @property
    def failed(self) -> int:
        return sum(case.status == "FAIL" for case in self.cases)

    @property
    def skipped(self) -> int:
        return sum(case.status == "SKIP" for case in self.cases)


def pytest_collection_modifyitems(items):
    """Attach case metadata before pytest creates its JUnit test reports."""
    for item in items:
        case_marker = item.get_closest_marker("case")
        if case_marker is None:
            continue

        parameter_id = getattr(getattr(item, "callspec", None), "id", "")
        case_id = str(case_marker.kwargs.get("case_id", item.nodeid))
        if parameter_id.startswith("TC-"):
            case_id = parameter_id

        category = next(
            (
                name
                for name in _CATEGORY_MARKERS
                if item.get_closest_marker(name) is not None
            ),
            "Regression Testing",
        )
        metadata = {
            "case_id": case_id,
            "category": next(
                (
                    label
                    for label, marker in CATEGORY_MARKERS.items()
                    if marker == category
                ),
                category,
            ),
            "description": str(case_marker.kwargs.get("description", item.name)),
            "expected_result": str(
                case_marker.kwargs.get(
                    "expected_result",
                    "All pytest assertions pass without failure.",
                )
            ),
        }
        item.user_properties.extend(metadata.items())


def parse_junit_xml(
    junit_xml: str,
    default_category: str = "Regression Testing",
) -> list[TestCaseResult]:
    """Convert pytest's JUnit XML into rows suitable for the dashboard."""
    root = ET.fromstring(junit_xml)
    results = []

    for testcase in root.iter("testcase"):
        properties = {
            prop.get("name", ""): prop.get("value", "")
            for prop in testcase.findall("./properties/property")
        }
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")
        issue = failure if failure is not None else error

        if issue is not None:
            status = "FAIL"
            failure_details = "\n".join(issue.itertext()).strip()
            actual_result = issue.get("message") or (
                failure_details.splitlines() or ["Pytest reported a failure."]
            )[0]
        elif skipped is not None:
            status = "SKIP"
            failure_details = "\n".join(skipped.itertext()).strip()
            actual_result = skipped.get("message") or "Pytest skipped this case."
        else:
            status = "PASS"
            failure_details = ""
            actual_result = "Pytest completed this case without failure."

        name = testcase.get("name", "unnamed test")
        classname = testcase.get("classname", "")
        case_id = properties.get("case_id") or f"{classname}::{name}".lstrip(":")
        description = properties.get("description") or name.replace("_", " ")
        category = properties.get("category") or default_category
        expected_result = properties.get(
            "expected_result", "All pytest assertions pass without failure."
        )

        results.append(
            TestCaseResult(
                case_id=case_id,
                category=category,
                description=description,
                expected_result=expected_result,
                actual_result=actual_result,
                status=status,
                duration_seconds=float(testcase.get("time", "0")),
                failure_details=failure_details,
            )
        )

    return results


def run_pytest_suite(category: str) -> TestRunResult:
    """Run pytest for one category or the full regression suite."""
    if category not in CATEGORY_MARKERS:
        raise ValueError(f"Unknown test category: {category}")

    marker = CATEGORY_MARKERS[category]
    with tempfile.TemporaryDirectory(prefix="software-testing-center-") as temp_dir:
        temp_path = Path(temp_dir)
        junit_path = temp_path / "pytest-results.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "testing_center",
            "--junitxml",
            str(junit_path),
            "--basetemp",
            str(temp_path / "pytest-tmp"),
        ]
        if marker is not None:
            command.extend(["-m", marker])
        command.append(str(PROJECT_ROOT / "tests"))

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TestExecutionError(
                "Pytest exceeded the 180-second execution limit.\n"
                f"Output: {error.stdout or ''}\n{error.stderr or ''}"
            ) from error
        except OSError as error:
            raise TestExecutionError(f"Could not start pytest: {error}") from error
        duration = time.perf_counter() - started

        if not junit_path.is_file():
            output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
            raise TestExecutionError(
                f"Pytest exited with code {completed.returncode} without a JUnit report.\n{output}"
            )

        junit_text = junit_path.read_text(encoding="utf-8")
        try:
            cases = parse_junit_xml(junit_text, default_category=category)
        except ET.ParseError as error:
            raise TestExecutionError(f"Pytest produced invalid JUnit XML: {error}") from error

        return TestRunResult(
            selected_category=category,
            exit_code=completed.returncode,
            duration_seconds=duration,
            cases=cases,
            junit_xml=junit_text,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def to_csv_report(result: TestRunResult) -> str:
    """Serialize actual case results as a downloadable CSV report."""
    output = io.StringIO(newline="")
    fieldnames = list(asdict(result.cases[0]).keys()) if result.cases else [
        "case_id",
        "category",
        "description",
        "expected_result",
        "actual_result",
        "status",
        "duration_seconds",
        "failure_details",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(asdict(case) for case in result.cases)
    return output.getvalue()


def to_json_report(result: TestRunResult) -> str:
    """Serialize actual suite and case results as downloadable JSON."""
    payload = {
        "selected_category": result.selected_category,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "total": len(result.cases),
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "cases": [asdict(case) for case in result.cases],
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return json.dumps(payload, indent=2)
