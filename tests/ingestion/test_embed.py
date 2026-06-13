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


import numpy as np

from backend.ingestion.chunker import Chunk
from backend.ingestion.embed import TABLE_NAME, _upsert_rows


def _make_chunk(
    chunk_id: str = "testchunk00000001",
    bbox: tuple | None = (0.0, 0.0, 595.0, 842.0),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        pdf_id="testpdf123456789",
        filename="doc.pdf",
        page_number=1,
        text="This is some chunk text.",
        token_count=5,
        language="en",
        bbox=bbox,
    )


def _make_upsert_client() -> MagicMock:
    mock = MagicMock()
    mock.table.return_value.upsert.return_value.execute.return_value = MagicMock()
    return mock


def test_upsert_rows_calls_table_upsert_execute():
    chunk = _make_chunk()
    vectors = np.zeros((1, 384))
    client = _make_upsert_client()
    _upsert_rows([chunk], vectors, client)
    client.table.assert_called_once_with(TABLE_NAME)
    client.table.return_value.upsert.assert_called_once()
    client.table.return_value.upsert.return_value.execute.assert_called_once()


def test_upsert_rows_row_contains_all_chunk_fields():
    chunk = _make_chunk()
    vectors = np.ones((1, 384))
    client = _make_upsert_client()
    _upsert_rows([chunk], vectors, client)
    rows = client.table.return_value.upsert.call_args[0][0]
    assert len(rows) == 1
    row = rows[0]
    assert row["chunk_id"] == chunk.chunk_id
    assert row["pdf_id"] == chunk.pdf_id
    assert row["filename"] == chunk.filename
    assert row["page_number"] == chunk.page_number
    assert row["text"] == chunk.text
    assert row["token_count"] == chunk.token_count
    assert row["language"] == chunk.language


def test_upsert_rows_embedding_is_list_of_floats():
    chunk = _make_chunk()
    vectors = np.ones((1, 384)) * 0.5
    client = _make_upsert_client()
    _upsert_rows([chunk], vectors, client)
    rows = client.table.return_value.upsert.call_args[0][0]
    emb = rows[0]["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(v, float) for v in emb)


def test_upsert_rows_bbox_tuple_stored_as_list():
    chunk = _make_chunk(bbox=(0.0, 0.0, 595.0, 842.0))
    vectors = np.zeros((1, 384))
    client = _make_upsert_client()
    _upsert_rows([chunk], vectors, client)
    rows = client.table.return_value.upsert.call_args[0][0]
    assert rows[0]["bbox"] == [0.0, 0.0, 595.0, 842.0]


def test_upsert_rows_none_bbox_stored_as_none():
    chunk = _make_chunk(bbox=None)
    vectors = np.zeros((1, 384))
    client = _make_upsert_client()
    _upsert_rows([chunk], vectors, client)
    rows = client.table.return_value.upsert.call_args[0][0]
    assert rows[0]["bbox"] is None


def test_upsert_rows_multiple_chunks():
    chunks = [_make_chunk(f"chunk{i:016d}") for i in range(3)]
    vectors = np.zeros((3, 384))
    client = _make_upsert_client()
    _upsert_rows(chunks, vectors, client)
    rows = client.table.return_value.upsert.call_args[0][0]
    assert len(rows) == 3
    assert rows[0]["chunk_id"] == "chunk0000000000000000"
    assert rows[2]["chunk_id"] == "chunk0000000000000002"
