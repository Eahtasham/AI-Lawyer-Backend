from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.logger import logger
import time

app = FastAPI(title="Legal RAG API", version="1.5.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    auth_header = request.headers.get("Authorization")
    auth_status = "Auth: YES" if auth_header else "Auth: NO"
    logger.info(f"➡️  {request.method} {request.url.path} [{auth_status}]")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | Time: {process_time:.2f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f" {request.method} {request.url.path} | "
            f"Error: {str(e)} | Time: {process_time:.2f}s"
        )
        raise

# Include routers
app.include_router(chat_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info("Legal RAG API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Legal RAG API shutting down...")

@app.get("/")
async def root():
    return {"message": "Legal RAG API is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}