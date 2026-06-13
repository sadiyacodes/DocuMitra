from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.ingestion.embed import EMBEDDING_MODEL, TABLE_NAME, _get_model


def test_constants_defined():
    assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert TABLE_NAME == "chunks"


def test_get_model_returns_sentence_transformer():
    with patch("backend.ingestion.embed.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        import backend.ingestion.embed as embed_mod
        embed_mod._model = None
        result = _get_model()
    mock_st.assert_called_once_with(EMBEDDING_MODEL)
    assert result is mock_st.return_value


def test_get_model_singleton_cached():
    with patch("backend.ingestion.embed.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()
        import backend.ingestion.embed as embed_mod
        embed_mod._model = None
        r1 = _get_model()
        r2 = _get_model()
    assert r1 is r2
    mock_st.assert_called_once()


from backend.ingestion.embed import TABLE_NAME, _fetch_existing_ids


def _make_client(existing: list[str]) -> MagicMock:
    """Mock Supabase client returning given chunk_ids from SELECT."""
    mock = MagicMock()
    mock.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [
        {"chunk_id": cid} for cid in existing
    ]
    return mock


def test_fetch_existing_ids_returns_matching_set():
    client = _make_client(["abc", "def"])
    result = _fetch_existing_ids({"abc", "def", "ghi"}, client)
    assert result == {"abc", "def"}


def test_fetch_existing_ids_empty_table_returns_empty_set():
    client = _make_client([])
    result = _fetch_existing_ids({"abc", "def"}, client)
    assert result == set()


def test_fetch_existing_ids_empty_input_returns_empty_set():
    client = _make_client([])
    result = _fetch_existing_ids(set(), client)
    assert result == set()


def test_fetch_existing_ids_queries_correct_table():
    client = _make_client([])
    _fetch_existing_ids({"abc"}, client)
    client.table.assert_called_once_with(TABLE_NAME)


def test_fetch_existing_ids_selects_chunk_id_column():
    client = _make_client([])
    _fetch_existing_ids({"abc"}, client)
    client.table.return_value.select.assert_called_once_with("chunk_id")
