"""Tests for keyword-based query routing."""
from backend.retrieval.router import route_query


def test_audit_log_query_routes_to_json():
    result = route_query("show me audit logs for failed login events")
    assert "json" in result


def test_employee_salary_query_routes_to_csv():
    result = route_query("what is the salary of employees in engineering department")
    assert "csv" in result


def test_policy_query_routes_to_pdf():
    result = route_query("what does the compliance policy say about data retention")
    assert "pdf" in result


def test_ambiguous_query_returns_all_three_types():
    result = route_query("give me a summary of everything")
    assert set(result) == {"pdf", "csv", "json"}


def test_result_has_no_duplicates():
    result = route_query("employee compliance policy audit log")
    assert len(result) == len(set(result))


def test_result_only_contains_valid_source_types():
    result = route_query("any query at all")
    assert all(t in {"pdf", "csv", "json"} for t in result)


def test_empty_query_returns_all_types():
    result = route_query("")
    assert set(result) == {"pdf", "csv", "json"}
