-- OliveSoft RAG schema for all-MiniLM-L6-v2 (384 dimensions).
-- Run this in a fresh Supabase project or after sql/migrate_1536_to_384.sql.

create extension if not exists vector;

create table if not exists public.knowledge_chunks (
    id uuid primary key,
    content text not null check (length(btrim(content)) > 0),
    doc_type text not null check (doc_type in ('cv', 'project', 'tool')),
    source_id text not null,
    source_file text not null,
    chunk_index integer not null check (chunk_index >= 0),
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(384) not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (source_id, chunk_index)
);

create index if not exists knowledge_chunks_embedding_idx
    on public.knowledge_chunks
    using hnsw (embedding vector_cosine_ops);

create index if not exists knowledge_chunks_source_idx
    on public.knowledge_chunks (source_id);

create or replace function public.match_knowledge_chunks (
    query_embedding vector(384),
    match_count integer default 5,
    filter_doc_type text default null,
    match_threshold double precision default 0.0
)
returns table (
    id uuid,
    content text,
    doc_type text,
    source_file text,
    metadata jsonb,
    similarity double precision
)
language sql
stable
security invoker
set search_path = public
as $$
    select
        kc.id,
        kc.content,
        kc.doc_type,
        kc.source_file,
        kc.metadata,
        1 - (kc.embedding <=> query_embedding) as similarity
    from public.knowledge_chunks as kc
    where
        (filter_doc_type is null or kc.doc_type = filter_doc_type)
        and (1 - (kc.embedding <=> query_embedding)) >= greatest(0.0, least(match_threshold, 1.0))
    order by kc.embedding <=> query_embedding
    limit greatest(1, least(match_count, 80));
$$;

-- The backend uses a service-role key and the HTTP service has its own token
-- check. Enabling RLS prevents accidental direct reads with Supabase anon keys.
alter table public.knowledge_chunks enable row level security;

revoke all on function public.match_knowledge_chunks(vector, integer, text, double precision)
    from public, anon, authenticated;
grant execute on function public.match_knowledge_chunks(vector, integer, text, double precision)
    to service_role;
