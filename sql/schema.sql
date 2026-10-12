-- RAG knowledge base schema: knowledge_chunks + match_knowledge_chunks.
--
-- DESTRUCTIVE: this drops and recreates the two RAG objects, deleting every
-- stored chunk. That is intended -- it is the from-scratch reset. Re-populate
-- afterwards with:
--   python -m rag.ingest_json --file "Cvs dataset/fake_cvs.json" --doc_type cv
--   python -m rag.ingest_json --file "Cvs dataset/fake_projects.json" --doc_type project
--
-- Run it in Supabase > SQL Editor > New query. The Python client cannot run it:
-- it talks to the REST layer, which can read and write rows but not create
-- tables, functions or extensions.
--
-- Only these two objects are dropped, never the schema, so anything else in
-- the project is left alone.

-- pgvector. Supabase installs extensions into the "extensions" schema; an
-- existing install in another schema is left where it is, which is why the
-- vector type and operators below are left unqualified and resolved through
-- the search path (the SQL Editor's includes both public and extensions).
create extension if not exists vector with schema extensions;

-- Every overload is dropped, not one signature: the previous schema defined a
-- 3-argument vector(1536) version, and create-or-replace cannot change a
-- function's argument list, so it would otherwise linger beside the new one.
do $$
declare
    fn regprocedure;
begin
    for fn in
        select p.oid::regprocedure
        from pg_proc p
        where p.proname = 'match_knowledge_chunks'
          and p.pronamespace = 'public'::regnamespace
    loop
        execute format('drop function %s', fn);
    end loop;
end $$;

drop table if exists public.knowledge_chunks cascade;

-- Columns mirror what rag/storage.py:build_rows writes, one for one.
-- id is not defaulted: make_chunk_id derives it from (source_id, chunk_index),
-- which is what makes re-running an ingest an idempotent upsert.
create table public.knowledge_chunks (
    id          uuid primary key,
    content     text not null,
    doc_type    text not null check (doc_type in ('cv', 'project', 'tool')),
    source_id   text not null,
    source_file text not null,
    chunk_index integer not null check (chunk_index >= 0),
    metadata    jsonb not null default '{}'::jsonb,
    -- 384 = sentence-transformers/all-MiniLM-L6-v2, rag/embeddings.py
    -- EMBEDDING_DIMENSIONS. The old schema used 1536 (Gemini); changing the
    -- embedding model means changing this and re-ingesting everything.
    embedding   vector(384) not null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- replace_source_chunks selects every id for a source before upserting, then
-- deletes the stale ones; without this index each ingest scans the table.
create index knowledge_chunks_source_id_idx
    on public.knowledge_chunks (source_id);

-- Embeddings are L2-normalised (normalize_embeddings=True), so cosine distance
-- is the right operator class.
create index knowledge_chunks_embedding_idx
    on public.knowledge_chunks
    using hnsw (embedding vector_cosine_ops);

-- Row Level Security on, with no policies. The service key the RAG service
-- uses bypasses RLS, so ingest and query work unchanged; the public anon key,
-- which ships in client apps, can no longer read or wipe the knowledge base.
alter table public.knowledge_chunks enable row level security;

-- Parameter names and the returned columns are the contract rag/query.py
-- validates: it calls this RPC by name with named arguments and rejects any
-- row missing one of these fields.
create function public.match_knowledge_chunks (
    query_embedding vector(384),
    match_count     integer default 5,
    filter_doc_type text default null,
    match_threshold double precision default 0
)
returns table (
    id          uuid,
    content     text,
    doc_type    text,
    source_file text,
    metadata    jsonb,
    similarity  double precision
)
language sql
stable
-- Pinned so the <=> operator resolves to pgvector and a caller cannot
-- redirect it through their own search_path.
set search_path = public, extensions
as $$
    select
        kc.id,
        kc.content,
        kc.doc_type,
        kc.source_file,
        kc.metadata,
        1 - (kc.embedding <=> query_embedding) as similarity
    from public.knowledge_chunks kc
    where (filter_doc_type is null or kc.doc_type = filter_doc_type)
      and 1 - (kc.embedding <=> query_embedding) >= match_threshold
    order by kc.embedding <=> query_embedding
    limit match_count;
$$;
