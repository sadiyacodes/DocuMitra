-- 1. Add new columns
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_id    text;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source_type  text NOT NULL DEFAULT 'pdf';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS access_roles text[] NOT NULL DEFAULT '{}';

-- 2. Backfill source_id from existing pdf_id values
UPDATE chunks SET source_id = pdf_id WHERE source_id IS NULL;

-- 3. Make source_id required
ALTER TABLE chunks ALTER COLUMN source_id SET NOT NULL;

-- 4. Drop the old match_chunks function (signature change requires DROP first)
DROP FUNCTION IF EXISTS match_chunks(vector, int);

-- 5. Create updated match_chunks RPC
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count     int,
  user_role       text    DEFAULT NULL,
  min_similarity  float   DEFAULT 0.4
)
RETURNS TABLE (
  chunk_id     text,
  source_id    text,
  source_type  text,
  filename     text,
  page_number  int,
  text         text,
  token_count  int,
  language     text,
  bbox         float[],
  access_roles text[],
  similarity   float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    chunk_id,
    source_id,
    source_type,
    filename,
    page_number,
    text,
    token_count,
    language,
    bbox,
    access_roles,
    1 - (embedding <=> query_embedding) AS similarity
  FROM chunks
  WHERE
    (user_role IS NULL OR access_roles @> ARRAY[user_role] OR access_roles = '{}')
    AND (1 - (embedding <=> query_embedding)) >= min_similarity
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
