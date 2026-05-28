import io
from app.logger import logger

class DocumentService:
    """Extracts text from PDF, DOCX, and TXT files."""

    def extract_text(self, file_bytes: bytes, file_type: str) -> str:
        """Route to the correct extractor based on file type."""
        extractors = {
            "pdf": self._extract_pdf,
            "docx": self._extract_docx,
            "txt": self._extract_txt,
        }
        extractor = extractors.get(file_type)
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_type}")
        text = extractor(file_bytes)
        # Remove null bytes — PostgreSQL text columns reject \u0000
        text = text.replace("\u0000", "")
        return text

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF. Falls back to OCR for scanned pages."""
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_pages = []
        ocr_needed_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text and len(text) > 30:
                text_pages.append(text)
            else:
                ocr_needed_pages.append(page_num)

        doc.close()

        # If we got enough text from PyMuPDF, return it
        if text_pages and len(ocr_needed_pages) <= len(text_pages):
            full_text = "\n\n".join(text_pages)
            logger.info(f"[DocService] Extracted {len(full_text)} chars from PDF (text-based, {len(text_pages)} pages)")
            return full_text

        # Fall back to OCR for scanned pages
        logger.info(f"[DocService] PDF appears scanned ({len(ocr_needed_pages)} pages need OCR). Attempting Tesseract...")
        return self._extract_pdf_ocr(file_bytes)

    def _extract_pdf_ocr(self, file_bytes: bytes) -> str:
        """OCR extraction for scanned PDFs using pdf2image + pytesseract."""
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(file_bytes, dpi=200)
            text_parts = []
            for i, img in enumerate(images):
                page_text = pytesseract.image_to_string(img, lang="eng")
                if page_text.strip():
                    text_parts.append(page_text.strip())

            if not text_parts:
                raise ValueError("OCR could not extract any text from the scanned PDF.")

            full_text = "\n\n".join(text_parts)
            logger.info(f"[DocService] OCR extracted {len(full_text)} chars from {len(images)} pages")
            return full_text

        except ImportError:
            logger.warning("[DocService] pytesseract/pdf2image not available, OCR disabled")
            raise ValueError("This PDF appears to be scanned/image-based. OCR is not available in this environment.")
        except Exception as e:
            logger.error(f"[DocService] OCR failed: {e}")
            raise ValueError(f"Failed to OCR scanned PDF: {str(e)}")

    def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX using python-docx."""
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        logger.info(f"[DocService] Extracted {len(full_text)} chars from DOCX ({len(paragraphs)} paragraphs)")
        return full_text

    def _extract_txt(self, file_bytes: bytes) -> str:
        """Extract text from plain text file."""
        full_text = file_bytes.decode("utf-8", errors="replace").strip()
        logger.info(f"[DocService] Extracted {len(full_text)} chars from TXT")
        return full_text

    def smart_truncate(self, text: str, max_chars: int = 20000) -> str:
        """Smart truncation: keeps beginning (definitions/parties) + end (obligations/penalties)."""
        if len(text) <= max_chars:
            return text
        
        head_size = int(max_chars * 0.6)  # 60% from beginning
        tail_size = int(max_chars * 0.4)  # 40% from end
        
        truncated = text[:head_size] + "\n\n[... document truncated for analysis ...]\n\n" + text[-tail_size:]
        logger.info(f"[DocService] Truncated document from {len(text)} to {len(truncated)} chars")
        return truncated

document_service = DocumentService()
