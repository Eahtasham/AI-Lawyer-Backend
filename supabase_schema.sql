-- 1. Profiles (Standard extension of auth.users)
create table if not exists public.profiles (
  id uuid not null references auth.users on delete cascade,
  full_name text,
  username text unique,
  avatar_url text,
  updated_at timestamp with time zone,
  primary key (id)
);

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger as $$
declare
  new_username text;
  base_username text;
  counter integer := 0;
begin
  -- Extract base username from email (before @)
  base_username := split_part(new.email, '@', 1);
  new_username := base_username;

  -- Check for uniqueness and append random digits if needed
  -- Simple loop to find a free username (usually succeeds 1st try or 2nd)
  loop
    if not exists (select 1 from public.profiles where username = new_username) then
       exit; -- unique found
    end if;
    
    -- If exists, append 4 random digits
    new_username := base_username || floor(random() * 9000 + 1000)::text;
  end loop;

  insert into public.profiles (id, full_name, username, avatar_url)
  values (new.id, new.raw_user_meta_data->>'full_name', new_username, new.raw_user_meta_data->>'avatar_url');
  return new;
end;
$$ language plpgsql security definer;

-- Trigger for new user creation
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();


-- 2. Conversations (Groups messages)
create table if not exists public.conversations (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  title text default 'New Conversation',
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  is_pinned boolean default false,
  is_deleted boolean default false not null
);

-- 3. Messages (Individual chat items)
create table if not exists public.messages (
  id uuid default gen_random_uuid() primary key,
  conversation_id uuid references public.conversations(id) on delete cascade not null,
  user_id uuid references auth.users(id) on delete cascade not null, -- Denial of Service protection
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  metadata jsonb, -- Stores citations, model info, specific tokens
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- RLS Policies
alter table profiles enable row level security;
alter table conversations enable row level security;
alter table messages enable row level security;

-- Profiles Policies
create policy "Users can view own profile" on profiles for select using (auth.uid() = id);
create policy "Users can update own profile" on profiles for update using (auth.uid() = id);

-- Conversations Policies
create policy "Users can view own conversations" on conversations for select using (auth.uid() = user_id and is_deleted = false);
create policy "Users can create own conversations" on conversations for insert with check (auth.uid() = user_id);
create policy "Users can update own conversations" on conversations for update using (auth.uid() = user_id);
create policy "Users can delete own conversations" on conversations for delete using (auth.uid() = user_id);

-- Messages Policies
create policy "Users can view own messages" on messages for select using (auth.uid() = user_id);
create policy "Users can insert own messages" on messages for insert with check (auth.uid() = user_id);

-- 4. Storage (Avatars)
-- Create a bucket for avatars if it doesn't exist
insert into storage.buckets (id, name, public)
values ('avatars', 'avatars', true)
on conflict (id) do nothing;

-- Storage Policies
-- Allow public access to view avatars
create policy "Avatar images are publicly accessible"
  on storage.objects for select
  using ( bucket_id = 'avatars' );

-- Allow authenticated users to upload their own avatar
create policy "Users can upload their own avatar"
  on storage.objects for insert
  with check (
    bucket_id = 'avatars' and
    auth.uid() = owner
  );

-- Allow users to update their own avatar
create policy "Users can update their own avatar"
  on storage.objects for update
  using (
    bucket_id = 'avatars' and
    auth.uid() = owner
  );
