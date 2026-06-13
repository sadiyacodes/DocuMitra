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
