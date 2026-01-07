from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.chat import router as chat_router
from app.logger import logger
import time

APP_NAME = "SamVidhaan API"
APP_VERSION = "1.7.2"
APP_DESCRIPTION = "An AI-supercharged assistant for understanding Indian law"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://samvidhaan.live",
        "https://dev.samvidhaan.live",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    auth_header = request.headers.get("Authorization")
    auth_status = "Auth: YES" if auth_header else "Auth: NO"
    logger.info(f"➡️  {request.method} {request.url.path} [{auth_status}]")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | Time: {process_time:.2f}s"
        )

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"{request.method} {request.url.path} | "
            f"Error: {str(e)} | Time: {process_time:.2f}s"
        )
        raise

# Routers
app.include_router(chat_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info(f"{APP_NAME} starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{APP_NAME} shutting down...")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/version")
async def get_version():
    return {"version": APP_VERSION}

@app.get("/health")
async def health():
    return {"status": "healthy"}
