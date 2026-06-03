-- Migration for RAG Evaluation and Benchmarking

-- Table to store per-query RAG metrics
CREATE TABLE IF NOT EXISTS rag_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL, -- Reference to the chat session
    message_id UUID, -- Reference to the specific message
    query TEXT NOT NULL,
    context_precision FLOAT CHECK (context_precision >= 0 AND context_precision <= 1),
    faithfulness FLOAT CHECK (faithfulness >= 0 AND faithfulness <= 1),
    relevance FLOAT CHECK (relevance >= 0 AND relevance <= 1),
    chunk_coverage FLOAT CHECK (chunk_coverage >= 0 AND chunk_coverage <= 1),
    model_used TEXT NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table to store aggregated benchmark reports
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    avg_relevance FLOAT,
    avg_faithfulness FLOAT,
    avg_context_precision FLOAT,
    avg_latency_ms INTEGER,
    total_queries INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for querying performance
CREATE INDEX IF NOT EXISTS idx_rag_metrics_session_id ON rag_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model ON benchmark_runs(model_name);
