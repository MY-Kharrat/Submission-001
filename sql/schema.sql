-- Run this in the Supabase SQL editor (Project > SQL Editor > New query)

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Table to hold every chunk of internal knowledge (CVs, projects, tools)
create table if not exists knowledge_chunks (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    doc_type text not null,        -- 'cv' | 'project' | 'tool'
    source_file text not null,     -- e.g. 'ahmed_cv.pdf'
    metadata jsonb default '{}',   -- anything extra: {"person": "Ahmed", "skills": [...]}
    embedding vector(1536),        -- 1536 = Gemini Embedding 2 output size, must match embeddings.py
    created_at timestamptz default now()
);

-- 3. Index for fast similarity search (HNSW is a good default)
create index if not exists knowledge_chunks_embedding_idx
    on knowledge_chunks
    using hnsw (embedding vector_cosine_ops);

-- 4. Function to query the nearest chunks to a given embedding
create or replace function match_knowledge_chunks (
    query_embedding vector(1536),
    match_count int default 5,
    filter_doc_type text default null
)
returns table (
    id uuid,
    content text,
    doc_type text,
    source_file text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        id,
        content,
        doc_type,
        source_file,
        metadata,
        1 - (embedding <=> query_embedding) as similarity
    from knowledge_chunks
    where filter_doc_type is null or doc_type = filter_doc_type
    order by embedding <=> query_embedding
    limit match_count;
$$;