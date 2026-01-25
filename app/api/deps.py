
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.db import db_service
from app.logger import logger

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verifies the Supabase JWT token and returns the user_id.
    """
    token = credentials.credentials
    logger.info(f"[Auth] Verifying token: {token[:10]}...")
    try:
        # We use the Supabase client to get the user.
        # This verifies the signature and expiration automatically.
        user = db_service.supabase.auth.get_user(token)
        
        if not user or not user.user:
             raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user.user.id
        
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
