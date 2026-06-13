-- DocuMitra — Supabase migration
-- Run this once in the Supabase SQL Editor (https://supabase.com/dashboard/project/_/sql)
-- Order matters: extension → table → index → function

-- ── 1. Enable pgvector extension ─────────────────────────────────────────────
create extension if not exists vector;

-- ── 2. Create chunks table ────────────────────────────────────────────────────
create table if not exists chunks (
    chunk_id    text        primary key,
    pdf_id      text        not null,
    filename    text        not null,
    page_number integer     not null,
    text        text        not null,
    token_count integer     not null,
    language    text        not null,
    bbox        jsonb,                         -- [x0, y0, x1, y1] or null
    embedding   vector(384) not null
);

-- ── 3. HNSW index for fast cosine similarity search ──────────────────────────
-- m=16 and ef_construction=64 are safe defaults; raise ef_construction for
-- higher recall at the cost of slower build time.
create index if not exists chunks_embedding_hnsw
    on chunks
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- ── 4. match_chunks RPC function ─────────────────────────────────────────────
-- Called by vector_store.py: client.rpc("match_chunks", {"query_embedding": [...], "match_count": k})
-- Returns cosine similarity = 1 - cosine_distance (embeddings are L2-normalised,
-- so this equals dot product, but the formula works for any normalised vectors).
create or replace function match_chunks(
    query_embedding vector(384),
    match_count     integer
)
returns table (
    chunk_id    text,
    pdf_id      text,
    filename    text,
    page_number integer,
    text        text,
    token_count integer,
    language    text,
    bbox        jsonb,
    similarity  float
)
language sql stable
as $$
    select
        c.chunk_id,
        c.pdf_id,
        c.filename,
        c.page_number,
        c.text,
        c.token_count,
        c.language,
        c.bbox,
        1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- ── 5. Row Level Security ─────────────────────────────────────────────────────
-- The backend uses the service_role key (bypasses RLS automatically).
-- If you use the anon key instead, enable RLS and add a permissive policy:
--
-- alter table chunks enable row level security;
-- create policy "allow all for service role"
--     on chunks for all
--     using (true);
