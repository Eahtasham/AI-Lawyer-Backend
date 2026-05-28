
from supabase import create_client, Client
from app.config import settings
from app.logger import logger
from typing import Optional, Dict, List
import datetime

class DatabaseService:
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL, 
            settings.SUPABASE_SERVICE_KEY
        )
    
    def create_conversation(self, user_id: str, title: str = "New Conversation", id: str = None) -> str:
        """Creates a new conversation and returns its ID."""
        try:
            data = {
                "user_id": user_id,
                "title": title
            }
            if id:
                data["id"] = id
                
            response = self.supabase.table("conversations").insert(data).execute()
            if response.data:
                return response.data[0]['id']
            raise Exception("Failed to create conversation")
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise

    def add_message(self, conversation_id: str, user_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Adds a message to a conversation."""
        try:
            data = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": role, # 'user' or 'assistant'
                "content": content,
                "metadata": metadata or {}
            }
            response = self.supabase.table("messages").insert(data).execute()
            if response.data:
                return response.data[0]['id']
            raise Exception("Failed to add message")
        except Exception as e:
            # Suppress logging for expected Foreign Key errors (e.g. when chat.py needs to create a conversation)
            if "23503" in str(e) or "foreign key constraint" in str(e).lower():
                raise
            logger.error(f"Error adding message: {e}")
            raise

    def delete_message(self, message_id: str) -> bool:
        """Hard deletes a specific message."""
        try:
            response = self.supabase.table("messages").delete().eq("id", message_id).execute()
            if response.data:
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            return False

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Soft deletes a conversation by setting is_deleted to True."""
        try:
             # Soft delete: set is_deleted = True instead of actually deleting
             response = self.supabase.table("conversations")\
                .update({"is_deleted": True})\
                .eq("id", conversation_id)\
                .eq("user_id", user_id)\
                .execute()
             
             # Supabase returns the updated rows. If len > 0, success.
             if response.data:
                 return True
             return False
        except Exception as e:
            logger.error(f"Error soft-deleting conversation: {e}")
            raise

    def update_conversation(self, conversation_id: str, user_id: str, title: str = None, is_pinned: bool = None) -> bool:
        """Updates a conversation's title or pinned status."""
        try:
            data = {}
            if title is not None:
                data["title"] = title
            if is_pinned is not None:
                data["is_pinned"] = is_pinned
            
            if not data:
                return False

            data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            response = self.supabase.table("conversations")\
                .update(data)\
                .eq("id", conversation_id)\
                .eq("user_id", user_id)\
                .execute()
            
            if response.data:
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating conversation: {e}")
            raise

    def get_conversation_history(self, conversation_id: str, user_id: str, limit: int = 50) -> List[Dict]:
        """Retrieves conversation history, ensuring user owns it."""
        try:
            # First verify ownership (optional if RLS is trusted, but good for backend logic)
            # We fetch messages ordered by created_at DESC (newest first) to get the LAST N messages.
            
            # Fetch messages
            response = self.supabase.table("messages")\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            
            # Return reversed list (Chronological order: Oldest -> Newest)
            if response.data:
                return response.data[::-1]
            return []
        except Exception as e:
            logger.error(f"Error fetching history: {e}")
            return []

    # --- Document Analysis Methods ---

    def create_document_analysis(self, user_id: str, file_name: str, file_type: str, file_size: int, storage_path: str) -> str:
        """Creates a new document analysis record and returns its ID."""
        try:
            data = {
                "user_id": user_id,
                "file_name": file_name,
                "file_type": file_type,
                "file_size": file_size,
                "storage_path": storage_path,
                "status": "pending",
            }
            response = self.supabase.table("document_analyses").insert(data).execute()
            if response.data:
                return response.data[0]["id"]
            raise Exception("Failed to create document analysis")
        except Exception as e:
            logger.error(f"Error creating document analysis: {e}")
            raise

    def update_analysis_status(self, analysis_id: str, status: str, analysis_json: dict = None, extracted_text: str = None):
        """Updates analysis status and optionally the analysis result and extracted text."""
        try:
            data = {
                "status": status,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            if analysis_json is not None:
                data["analysis"] = analysis_json
            if extracted_text is not None:
                data["extracted_text"] = extracted_text

            self.supabase.table("document_analyses").update(data).eq("id", analysis_id).execute()
        except Exception as e:
            logger.error(f"Error updating analysis status: {e}")
            raise

    def get_user_analyses(self, user_id: str) -> List[Dict]:
        """Returns list of user's document analyses for sidebar."""
        try:
            response = self.supabase.table("document_analyses")\
                .select("id, file_name, file_type, file_size, status, created_at")\
                .eq("user_id", user_id)\
                .eq("is_deleted", False)\
                .order("created_at", desc=True)\
                .execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error fetching analyses: {e}")
            return []

    def get_analysis(self, analysis_id: str, user_id: str) -> Optional[Dict]:
        """Returns a single analysis by ID, ensuring user owns it."""
        try:
            response = self.supabase.table("document_analyses")\
                .select("*")\
                .eq("id", analysis_id)\
                .eq("user_id", user_id)\
                .eq("is_deleted", False)\
                .single()\
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching analysis: {e}")
            return None

    def delete_analysis(self, analysis_id: str, user_id: str) -> bool:
        """Soft deletes a document analysis."""
        try:
            response = self.supabase.table("document_analyses")\
                .update({"is_deleted": True})\
                .eq("id", analysis_id)\
                .eq("user_id", user_id)\
                .execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error deleting analysis: {e}")
            return False

db_service = DatabaseService()
