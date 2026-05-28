import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from app.api.deps import get_current_user
from app.services.db import db_service
from app.services.document import document_service
from app.services.analyzer import analyzer_service
from app.config import settings
from app.logger import logger

router = APIRouter()

ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def _get_file_type(filename: str, content_type: str) -> str:
    """Determine file type from extension or content type."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ALLOWED_EXTENSIONS:
        return ext
    if content_type in ALLOWED_TYPES:
        return ALLOWED_TYPES[content_type]
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or content_type}. Allowed: PDF, DOCX, TXT")


@router.post("/document/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a document, extract text, create analysis record."""
    try:
        # Validate file type
        file_type = _get_file_type(file.filename, file.content_type)

        # Read file bytes
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # Validate size
        if file_size > settings.MAX_DOCUMENT_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {settings.MAX_DOCUMENT_SIZE // (1024*1024)}MB.",
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty.")

        # Generate storage path
        file_id = str(uuid.uuid4())
        storage_path = f"{user_id}/{file_id}_{file.filename}"

        # Upload to Supabase Storage
        try:
            db_service.supabase.storage.from_("documents").upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": file.content_type or "application/octet-stream"},
            )
        except Exception as e:
            logger.warning(f"[Upload] Storage upload failed (may not be configured): {e}")
            # Continue even if storage fails — we have the bytes in memory for analysis

        # Extract text
        extracted_text = document_service.extract_text(file_bytes, file_type)

        if not extracted_text or len(extracted_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful text from this document. It may be image-based or empty.",
            )

        # Create DB record
        analysis_id = db_service.create_document_analysis(
            user_id=user_id,
            file_name=file.filename,
            file_type=file_type,
            file_size=file_size,
            storage_path=storage_path,
        )

        # Store extracted text
        db_service.update_analysis_status(analysis_id, "pending", extracted_text=extracted_text)

        logger.info(f"[Upload] Document uploaded: {file.filename} ({file_size} bytes, {len(extracted_text)} chars text)")

        return {
            "analysis_id": analysis_id,
            "file_name": file.filename,
            "file_type": file_type,
            "file_size": file_size,
            "text_length": len(extracted_text),
            "status": "pending",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Upload] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/analyze/{analysis_id}")
async def analyze_document(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    """Stream analysis results for an uploaded document."""
    # Fetch the analysis record
    record = db_service.get_analysis(analysis_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    extracted_text = record.get("extracted_text")
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No extracted text found for this document")

    file_name = record.get("file_name", "document")

    async def event_generator():
        try:
            # Update status to processing
            db_service.update_analysis_status(analysis_id, "processing")

            final_sections = []
            final_chunks = []

            async for event in analyzer_service.analyze_document_stream(extracted_text, file_name):
                clean_event = event.strip()

                # Capture sections for DB storage
                if clean_event.startswith("section:"):
                    try:
                        section_data = json.loads(clean_event[8:].strip())
                        final_sections.append(section_data)
                    except Exception:
                        pass
                elif clean_event.startswith("chunks:"):
                    try:
                        final_chunks = json.loads(clean_event[7:].strip())
                    except Exception:
                        pass
                elif clean_event.startswith("done:"):
                    try:
                        done_data = json.loads(clean_event[5:].strip())
                        final_sections = done_data.get("sections", final_sections)
                    except Exception:
                        pass

                yield event

            # Save completed analysis to DB
            analysis_result = {
                "sections": final_sections,
                "chunks": final_chunks,
            }
            db_service.update_analysis_status(analysis_id, "completed", analysis_json=analysis_result)

        except Exception as e:
            logger.error(f"[Analyze] Error: {e}", exc_info=True)
            db_service.update_analysis_status(analysis_id, "failed")
            yield f"log: ✗ Analysis failed: {str(e)}\n"
            yield f"data: {json.dumps({'error': str(e)})}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/document/analyses")
async def list_analyses(
    user_id: str = Depends(get_current_user),
):
    """List all document analyses for the current user."""
    analyses = db_service.get_user_analyses(user_id)
    return {"analyses": analyses}


@router.get("/document/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get a single completed analysis."""
    record = db_service.get_analysis(analysis_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


@router.delete("/document/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    """Soft delete a document analysis."""
    success = db_service.delete_analysis(analysis_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found or access denied")
    return {"status": "success", "message": "Analysis deleted"}


@router.get("/document/{analysis_id}/followup")
async def followup_chat(
    analysis_id: str,
    question: str,
    user_id: str = Depends(get_current_user),
):
    """Stream a follow-up chat response about the analyzed document."""
    record = db_service.get_analysis(analysis_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    extracted_text = record.get("extracted_text", "")
    analysis_json = record.get("analysis", {})

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No document text available for follow-up")

    async def event_generator():
        try:
            async for event in analyzer_service.followup_chat_stream(
                extracted_text, analysis_json, question
            ):
                yield event
        except Exception as e:
            logger.error(f"[FollowUp] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
