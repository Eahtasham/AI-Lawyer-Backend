-- Document Analyses table for the Legal Document Analyzer feature
CREATE TABLE IF NOT EXISTS public.document_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    file_size INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    analysis JSONB DEFAULT NULL,
    extracted_text TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    is_deleted BOOLEAN DEFAULT false
);

-- Enable Row Level Security
ALTER TABLE public.document_analyses ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Users can only access their own analyses
CREATE POLICY "Users can view own analyses"
    ON public.document_analyses FOR SELECT
    USING (auth.uid() = user_id AND is_deleted = false);

CREATE POLICY "Users can insert own analyses"
    ON public.document_analyses FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own analyses"
    ON public.document_analyses FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own analyses"
    ON public.document_analyses FOR DELETE
    USING (auth.uid() = user_id);

-- Create Supabase Storage bucket for documents (run via Supabase dashboard or API)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('documents', 'documents', false);

-- Storage policies for the documents bucket
-- CREATE POLICY "Users can upload own documents"
--     ON storage.objects FOR INSERT
--     WITH CHECK (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

-- CREATE POLICY "Users can read own documents"
--     ON storage.objects FOR SELECT
--     USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

-- CREATE POLICY "Users can delete own documents"
--     ON storage.objects FOR DELETE
--     USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);
