-- Migration: Add soft delete support for conversations
-- Run this in your Supabase SQL Editor

-- 1. Add is_deleted column to conversations table
ALTER TABLE public.conversations
ADD COLUMN IF NOT EXISTS is_deleted boolean DEFAULT false NOT NULL;

-- 2. Drop old select policy
DROP POLICY IF EXISTS "Users can view own conversations" ON conversations;

-- 3. Create new select policy that excludes soft-deleted conversations
CREATE POLICY "Users can view own conversations" ON conversations 
  FOR SELECT USING (auth.uid() = user_id AND is_deleted = false);

-- 4. Add update policy if not exists (needed for soft delete to work)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE tablename = 'conversations' 
        AND policyname = 'Users can update own conversations'
    ) THEN
        CREATE POLICY "Users can update own conversations" ON conversations 
          FOR UPDATE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Verification query (run separately to check):
-- SELECT id, title, is_deleted, created_at FROM public.conversations LIMIT 10;
