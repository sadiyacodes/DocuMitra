from __future__ import annotations

from backend.retrieval.vector_store import SearchResult
from backend.generation.prompt_templates import NO_ANSWER_RESPONSE, SYSTEM_PROMPT, build_user_message


def _make_result(filename: str = "doc.pdf", page: int = 3, text: str = "Some text.") -> SearchResult:
    return SearchResult(
        chunk_id="abc123def456abcd",
        source_id="testpdf123456789",
        source_type="pdf",
        filename=filename,
        page_number=page,
        text=text,
        token_count=3,
        language="en",
        bbox=None,
        access_roles=[],
        similarity=0.9,
    )


def test_system_prompt_is_nonempty_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


def test_system_prompt_contains_citation_format():
    assert "[filename, p.N]" in SYSTEM_PROMPT


def test_system_prompt_contains_no_info_instruction():
    assert "not enough information" in SYSTEM_PROMPT


def test_system_prompt_prohibits_guessing():
    assert "not present in the excerpts" in SYSTEM_PROMPT


def test_build_user_message_includes_filename():
    result = _make_result(filename="annual_report.pdf")
    msg = build_user_message("query", [result])
    assert "annual_report.pdf" in msg


def test_build_user_message_includes_page_number():
    result = _make_result(page=7)
    msg = build_user_message("query", [result])
    assert "p.7" in msg


def test_build_user_message_includes_chunk_text():
    result = _make_result(text="Revenue grew 12% YoY.")
    msg = build_user_message("query", [result])
    assert "Revenue grew 12% YoY." in msg


def test_build_user_message_includes_query():
    result = _make_result()
    msg = build_user_message("What is the revenue?", [result])
    assert "What is the revenue?" in msg


def test_build_user_message_empty_results_still_includes_query():
    msg = build_user_message("What is the capital?", [])
    assert "What is the capital?" in msg


def test_build_user_message_multiple_results_all_included():
    results = [
        _make_result(filename="a.pdf", page=1, text="First chunk."),
        _make_result(filename="b.pdf", page=2, text="Second chunk."),
    ]
    msg = build_user_message("query", results)
    assert "a.pdf" in msg
    assert "b.pdf" in msg
    assert "First chunk." in msg
    assert "Second chunk." in msg


def test_no_answer_response_is_nonempty_string():
    assert isinstance(NO_ANSWER_RESPONSE, str)
    assert len(NO_ANSWER_RESPONSE) > 0


def test_no_answer_response_matches_system_prompt():
    assert "not enough information" in NO_ANSWER_RESPONSE
