"""Tests for Software Testing Center result parsing and report downloads."""

import json

import pytest

from testing_center import (
    TestRunResult as RunResult,
    parse_junit_xml,
    to_csv_report,
    to_json_report,
)


@pytest.mark.unit
@pytest.mark.case(
    case_id="TC-UNIT-JUNIT-PARSING",
    description="Parse actual pytest pass and failure records from JUnit XML",
    expected_result="Status, metadata, durations, and failure details match the XML.",
)
def test_junit_parser_extracts_case_results() -> None:
    report = """\
<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0">
  <testcase classname="tests.test_sample" name="test_pass" time="0.125">
    <properties>
      <property name="case_id" value="TC-UNIT-PASS" />
      <property name="category" value="Unit Testing" />
      <property name="description" value="Passing sample case" />
      <property name="expected_result" value="The assertion passes." />
    </properties>
  </testcase>
  <testcase classname="tests.test_sample" name="test_fail" time="0.250">
    <properties>
      <property name="case_id" value="TC-UNIT-FAIL" />
      <property name="category" value="Unit Testing" />
    </properties>
    <failure message="assert 1 == 2">AssertionError: assert 1 == 2</failure>
  </testcase>
</testsuite></testsuites>
"""

    cases = parse_junit_xml(report)

    assert [case.status for case in cases] == ["PASS", "FAIL"]
    assert cases[0].case_id == "TC-UNIT-PASS"
    assert cases[0].category == "Unit Testing"
    assert cases[0].duration_seconds == 0.125
    assert cases[1].case_id == "TC-UNIT-FAIL"
    assert cases[1].actual_result == "assert 1 == 2"
    assert "AssertionError" in cases[1].failure_details
    assert cases[1].duration_seconds == 0.25


@pytest.mark.unit
@pytest.mark.case(
    case_id="TC-UNIT-REPORT-DOWNLOAD",
    description="Serialize execution results into downloadable CSV and JSON reports",
    expected_result="Reports contain the actual run summary and test-case results.",
)
def test_reports_serialize_actual_results() -> None:
    cases = parse_junit_xml(
        '<testsuite><testcase classname="sample" name="test_one" time="0.1">'
        "</testcase></testsuite>"
    )
    result = RunResult(
        selected_category="Regression Testing",
        exit_code=0,
        duration_seconds=0.15,
        cases=cases,
        junit_xml="<testsuite />",
        stdout="1 passed",
        stderr="",
    )

    csv_report = to_csv_report(result)
    json_report = json.loads(to_json_report(result))

    assert "case_id,category,description" in csv_report
    assert cases[0].case_id in csv_report
    assert json_report["passed"] == 1
    assert json_report["failed"] == 0
    assert json_report["cases"][0]["status"] == "PASS"
