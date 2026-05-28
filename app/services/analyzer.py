import asyncio
import httpx
import json
from typing import List, Dict, AsyncGenerator
from app.config import settings
from app.logger import logger
from app.services.qdrant import qdrant_service


class AnalyzerService:
    """
    Two-pass document analysis service.
    Pass 1: Extract key legal topics from document (fast, small context).
    Pass 2: Query Qdrant + analyze document with retrieved legal context.
    """

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 5, 10]  # seconds

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def _call_gemini_with_retry(self, model: str, prompt: str, temperature: float = 0.3) -> str:
        """Non-streaming Gemini call with retry for 503/429 errors."""
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        try:
                            return data["candidates"][0]["content"]["parts"][0]["text"]
                        except (KeyError, IndexError):
                            logger.error(f"[Analyzer] Malformed Gemini response: {data}")
                            return ""

                    if response.status_code in (503, 429) and attempt < self.MAX_RETRIES - 1:
                        delay = self.RETRY_DELAYS[attempt]
                        logger.warning(f"[Analyzer] Gemini {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        await asyncio.sleep(delay)
                        continue

                    logger.error(f"[Analyzer] Gemini error ({response.status_code}): {response.text[:300]}")
                    response.raise_for_status()

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(f"[Analyzer] Request failed ({e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise
        return ""

    async def _stream_gemini_with_retry(self, model: str, prompt: str, temperature: float = 0.3) -> AsyncGenerator[str, None]:
        """Streaming Gemini call with retry for 503/429."""
        url = f"{self.base_url}/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code in (503, 429) and attempt < self.MAX_RETRIES - 1:
                            await response.aread()
                            delay = self.RETRY_DELAYS[attempt]
                            logger.warning(f"[Analyzer] Stream {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                            await asyncio.sleep(delay)
                            continue

                        if response.status_code != 200:
                            error_body = await response.aread()
                            logger.error(f"[Analyzer] Stream error ({response.status_code}): {error_body.decode()[:300]}")
                            return

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            try:
                                json_str = line[5:].strip()
                                if not json_str:
                                    continue
                                data = json.loads(json_str)
                                if "candidates" in data:
                                    candidate = data["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        text_chunk = candidate["content"]["parts"][0].get("text", "")
                                        if text_chunk:
                                            yield text_chunk
                            except Exception:
                                pass
                        return  # Success, exit retry loop

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(f"[Analyzer] Stream request failed ({e}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[Analyzer] Stream failed after {self.MAX_RETRIES} attempts: {e}")

    async def _extract_topics(self, document_text: str) -> List[str]:
        """Pass 1: Extract key legal topics from first 10K chars of document."""
        snippet = document_text[:10000]
        prompt = f"""You are a legal document analyst. Read the following document excerpt and extract 3-5 key legal topics, acts, sections, or legal concepts mentioned.

Return ONLY a JSON array of search query strings. Each query should be specific enough to find relevant Indian laws or case precedents.

Example output: ["Section 420 IPC fraud", "breach of contract Indian Contract Act", "specific performance"]

DOCUMENT EXCERPT:
{snippet}

JSON ARRAY:"""

        try:
            result = await self._call_gemini_with_retry(settings.MODEL_ANALYZER, prompt, temperature=0.2)
            # Parse JSON array from response
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            topics = json.loads(result)
            if isinstance(topics, list) and len(topics) > 0 and all(isinstance(t, str) for t in topics):
                logger.info(f"[Analyzer] Extracted topics: {topics}")
                return topics[:5]
            else:
                logger.warning(f"[Analyzer] Topic extraction returned empty or invalid: {result[:200]}")
        except Exception as e:
            logger.error(f"[Analyzer] Topic extraction failed: {e}")

        # Fallback: use first 200 chars as a single search query
        fallback = document_text[:200].replace("\n", " ").strip()
        logger.info(f"[Analyzer] Using fallback topic: {fallback[:80]}...")
        return [fallback]

    async def _retrieve_legal_context(self, topics: List[str]) -> tuple[List[Dict], List[Dict]]:
        """Query Qdrant with extracted topics. Returns (statute_chunks, case_chunks)."""
        loop = asyncio.get_running_loop()
        all_statute_chunks = []
        all_case_chunks = []
        seen_statutes = set()
        seen_cases = set()

        logger.info(f"[Analyzer] ========== QDRANT RETRIEVAL ==========")
        logger.info(f"[Analyzer] Topics to search: {topics}")

        for topic in topics[:3]:  # Max 3 queries to limit context
            try:
                logger.info(f"[Analyzer] Searching for topic: '{topic}'")

                statute_results = await loop.run_in_executor(
                    None, qdrant_service.search_statutes, topic, 3
                )
                case_results = await loop.run_in_executor(
                    None, qdrant_service.search_cases, topic, 3
                )

                logger.info(f"[Analyzer]   → Statutes: {len(statute_results)} results")
                logger.info(f"[Analyzer]   → Cases: {len(case_results)} results")

                for chunk in statute_results:
                    key = chunk.get("text", "")[:100]
                    if key not in seen_statutes:
                        seen_statutes.add(key)
                        all_statute_chunks.append(chunk)
                        meta = chunk.get("metadata", {})
                        logger.info(f"[Analyzer]     + Statute: {meta.get('law', 'Unknown')} - S.{meta.get('section_number', '?')} (score: {chunk.get('score', 0):.3f})")

                for chunk in case_results:
                    key = chunk.get("text", "")[:100]
                    if key not in seen_cases:
                        seen_cases.add(key)
                        all_case_chunks.append(chunk)
                        meta = chunk.get("metadata", {})
                        logger.info(f"[Analyzer]     + Case: {meta.get('case_name', 'Unknown')} (score: {chunk.get('score', 0):.3f})")

            except Exception as e:
                logger.error(f"[Analyzer] Qdrant search failed for topic '{topic}': {e}")

        # Limit to top 5 total each, re-rank by score
        all_statute_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_case_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(f"[Analyzer] Final retrieval: {len(all_statute_chunks[:5])} statutes, {len(all_case_chunks[:5])} cases")

        return all_statute_chunks[:5], all_case_chunks[:5]

    def _format_legal_context(self, statute_chunks: List[Dict], case_chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string, capped at 1500 chars each."""
        parts = []
        if statute_chunks:
            parts.append("=== RELEVANT STATUTES FROM DATABASE ===")
            for i, chunk in enumerate(statute_chunks, 1):
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "")[:1500]
                law = meta.get("law", "Unknown Act")
                section = meta.get("section_number", "")
                parts.append(f"\n[{i}] {law} - Section {section}\n{text}")

        if case_chunks:
            parts.append("\n=== RELEVANT CASE PRECEDENTS FROM DATABASE ===")
            for i, chunk in enumerate(case_chunks, 1):
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "")[:1500]
                case_name = meta.get("case_name", "Unknown Case")
                parts.append(f"\n[{i}] {case_name}\n{text}")

        return "\n".join(parts)

    async def analyze_document_stream(self, extracted_text: str, file_name: str) -> AsyncGenerator[str, None]:
        """
        Main analysis pipeline (streaming).
        Pass 1: Extract topics → Query Qdrant
        Pass 2: Analyze document with legal context → Stream sections
        """
        import time
        pipeline_start = time.time()

        logger.info(f"[Analyzer] ========== PIPELINE START ==========")
        logger.info(f"[Analyzer] Document: {file_name}")
        logger.info(f"[Analyzer] Text length: {len(extracted_text)} chars")
        logger.info(f"[Analyzer] Model: {settings.MODEL_ANALYZER}")

        yield f"log: Starting analysis of '{file_name}'...\n"
        yield f"log: Document size: {len(extracted_text):,} characters\n"

        # --- PASS 1: Topic Extraction ---
        step1_start = time.time()
        logger.info(f"[Analyzer] ========== STEP 1: TOPIC EXTRACTION ==========")
        logger.info(f"[Analyzer] Sending first {min(10000, len(extracted_text))} chars to {settings.MODEL_ANALYZER}")
        yield f"log: \n"
        yield f"log: [STEP 1/3] Extracting key legal topics from document...\n"
        yield f"log: Using model: {settings.MODEL_ANALYZER}\n"

        topics = await self._extract_topics(extracted_text)
        step1_time = time.time() - step1_start

        logger.info(f"[Analyzer] Step 1 completed in {step1_time:.1f}s")
        logger.info(f"[Analyzer] Topics found: {topics}")

        yield f"log: ✓ Found {len(topics)} legal topics ({step1_time:.1f}s)\n"
        for i, topic in enumerate(topics, 1):
            yield f"log:   {i}. {topic[:60]}{'...' if len(topic) > 60 else ''}\n"

        # --- PASS 1.5: Qdrant Retrieval ---
        step2_start = time.time()
        logger.info(f"[Analyzer] ========== STEP 2: DATABASE RETRIEVAL ==========")
        yield f"log: \n"
        yield f"log: [STEP 2/3] Searching legal databases for related laws & precedents...\n"
        yield f"log: → Querying Statutes database (Indian Penal Code, CrPC, etc.)\n"
        yield f"log: → Querying Case Law database (Supreme Court precedents)\n"

        statute_chunks, case_chunks = await self._retrieve_legal_context(topics)
        step2_time = time.time() - step2_start

        logger.info(f"[Analyzer] Step 2 completed in {step2_time:.1f}s")

        yield f"log: ✓ Found {len(statute_chunks)} statutes, {len(case_chunks)} case precedents ({step2_time:.1f}s)\n"

        # Log individual results
        if statute_chunks:
            for chunk in statute_chunks:
                meta = chunk.get("metadata", {})
                yield f"log:   📖 {meta.get('law', 'Unknown')} - S.{meta.get('section_number', '?')} (relevance: {chunk.get('score', 0):.0%})\n"
        if case_chunks:
            for chunk in case_chunks:
                meta = chunk.get("metadata", {})
                yield f"log:   ⚖️ {meta.get('case_name', 'Unknown Case')} (relevance: {chunk.get('score', 0):.0%})\n"

        # Send retrieved chunks to frontend
        all_chunks = statute_chunks + case_chunks
        for i, c in enumerate(all_chunks, 1):
            c["rank"] = i
        if all_chunks:
            yield f"chunks: {json.dumps(all_chunks)}\n"

        legal_context = self._format_legal_context(statute_chunks, case_chunks)
        total_context_chars = len(extracted_text[:20000]) + len(legal_context)
        logger.info(f"[Analyzer] Total context for analysis: {total_context_chars} chars")

        # --- PASS 2: Full Analysis ---
        step3_start = time.time()
        logger.info(f"[Analyzer] ========== STEP 3: GEMINI ANALYSIS ==========")
        logger.info(f"[Analyzer] Document context: {min(20000, len(extracted_text))} chars")
        logger.info(f"[Analyzer] Legal context: {len(legal_context)} chars")
        logger.info(f"[Analyzer] Total input: ~{total_context_chars} chars (~{total_context_chars // 4} tokens)")

        yield f"log: \n"
        yield f"log: [STEP 3/3] Generating comprehensive legal analysis...\n"
        yield f"log: Context size: {total_context_chars:,} characters (~{total_context_chars // 4:,} tokens)\n"
        yield f"log: Streaming analysis from {settings.MODEL_ANALYZER}...\n"

        truncated_text = extracted_text[:20000] if len(extracted_text) > 20000 else extracted_text
        if len(extracted_text) > 20000:
            head = extracted_text[:12000]
            tail = extracted_text[-8000:]
            truncated_text = head + "\n\n[... document truncated ...]\n\n" + tail
            yield f"log: Document truncated from {len(extracted_text):,} to {len(truncated_text):,} chars (smart truncation)\n"

        prompt = f"""{settings.PROMPT_ANALYZER}

DOCUMENT ({file_name}):
\"\"\"
{truncated_text}
\"\"\"

{legal_context}

Analyze the document and provide your response as a valid JSON object with the following structure:
{{
  "summary": "A comprehensive 2-3 paragraph summary of the document",
  "key_clauses": [
    {{"text": "clause text or reference", "significance": "why this clause matters"}}
  ],
  "risk_analysis": [
    {{"text": "risk description", "severity": "high|medium|low"}}
  ],
  "obligations": [
    {{"party": "who", "text": "what they must do", "deadline": "when, if specified"}}
  ],
  "legal_jargon": [
    {{"term": "legal term used", "simplified": "plain language explanation"}}
  ],
  "related_laws": "A paragraph describing how the related statutes and case precedents from the database connect to this document"
}}

Return ONLY the JSON object, no markdown fences."""

        # Collect full response then parse sections
        full_response = []
        async for chunk in self._stream_gemini_with_retry(
            settings.MODEL_ANALYZER, prompt, temperature=settings.TEMPERATURE_ANALYZER
        ):
            full_response.append(chunk)

        raw_text = "".join(full_response)
        step3_time = time.time() - step3_start

        logger.info(f"[Analyzer] Step 3 completed in {step3_time:.1f}s")
        logger.info(f"[Analyzer] Raw response length: {len(raw_text)} chars")

        # Parse JSON response
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            analysis = json.loads(cleaned)
            logger.info(f"[Analyzer] JSON parsed successfully. Keys: {list(analysis.keys())}")

            sections = []

            # 1. Summary
            if analysis.get("summary"):
                section = {"title": "Summary", "content": analysis["summary"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                yield f"log: ✓ Summary generated\n"
                logger.info(f"[Analyzer] Section: Summary ({len(analysis['summary'])} chars)")

            # 2. Key Clauses
            if analysis.get("key_clauses"):
                section = {"title": "Key Clauses", "items": analysis["key_clauses"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                yield f"log: ✓ Identified {len(analysis['key_clauses'])} key clauses\n"
                logger.info(f"[Analyzer] Section: Key Clauses ({len(analysis['key_clauses'])} items)")

            # 3. Risk Analysis
            if analysis.get("risk_analysis"):
                section = {"title": "Risk Analysis", "items": analysis["risk_analysis"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                high = sum(1 for r in analysis["risk_analysis"] if r.get("severity", "").lower() == "high")
                med = sum(1 for r in analysis["risk_analysis"] if r.get("severity", "").lower() == "medium")
                low = sum(1 for r in analysis["risk_analysis"] if r.get("severity", "").lower() == "low")
                yield f"log: ✓ Found {len(analysis['risk_analysis'])} risks ({high} high, {med} medium, {low} low)\n"
                logger.info(f"[Analyzer] Section: Risk Analysis ({len(analysis['risk_analysis'])} items — {high}H/{med}M/{low}L)")

            # 4. Obligations
            if analysis.get("obligations"):
                section = {"title": "Obligations & Deadlines", "items": analysis["obligations"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                yield f"log: ✓ Extracted {len(analysis['obligations'])} obligations\n"
                logger.info(f"[Analyzer] Section: Obligations ({len(analysis['obligations'])} items)")

            # 5. Legal Jargon
            if analysis.get("legal_jargon"):
                section = {"title": "Legal Jargon Simplified", "items": analysis["legal_jargon"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                yield f"log: ✓ Simplified {len(analysis['legal_jargon'])} legal terms\n"
                logger.info(f"[Analyzer] Section: Legal Jargon ({len(analysis['legal_jargon'])} items)")

            # 6. Related Laws (from Qdrant)
            if analysis.get("related_laws"):
                section = {"title": "Related Laws & Precedents", "content": analysis["related_laws"]}
                sections.append(section)
                yield f"section: {json.dumps(section)}\n"
                yield f"log: ✓ Cross-referenced with legal database\n"
                logger.info(f"[Analyzer] Section: Related Laws ({len(analysis['related_laws'])} chars)")

            total_time = time.time() - pipeline_start
            logger.info(f"[Analyzer] ========== PIPELINE COMPLETE ==========")
            logger.info(f"[Analyzer] Total time: {total_time:.1f}s (Topic: {step1_time:.1f}s, Retrieval: {step2_time:.1f}s, Analysis: {step3_time:.1f}s)")
            logger.info(f"[Analyzer] Sections generated: {len(sections)}")

            yield f"done: {json.dumps({'sections': sections})}\n"
            yield f"log: \n"
            yield f"log: ✓ Analysis complete — {len(sections)} sections in {total_time:.1f}s\n"
            yield f"log:   Step 1 (Topics): {step1_time:.1f}s\n"
            yield f"log:   Step 2 (Database): {step2_time:.1f}s\n"
            yield f"log:   Step 3 (Analysis): {step3_time:.1f}s\n"

        except json.JSONDecodeError as e:
            logger.error(f"[Analyzer] Failed to parse analysis JSON: {e}")
            logger.error(f"[Analyzer] Raw response preview: {raw_text[:500]}...")
            # Fallback: return raw text as summary
            section = {"title": "Summary", "content": raw_text}
            yield f"section: {json.dumps(section)}\n"
            yield f"done: {json.dumps({'sections': [section]})}\n"
            yield f"log: ⚠ Analysis completed with parsing fallback\n"

    async def followup_chat_stream(
        self, extracted_text: str, analysis_json: dict, question: str
    ) -> AsyncGenerator[str, None]:
        """Stream a follow-up answer about the analyzed document."""
        logger.info(f"[Analyzer] ========== FOLLOW-UP CHAT ==========")
        logger.info(f"[Analyzer] Question: {question}")
        logger.info(f"[Analyzer] Document context: {min(15000, len(extracted_text))} chars")

        doc_context = extracted_text[:15000] if len(extracted_text) > 15000 else extracted_text
        analysis_summary = json.dumps(analysis_json, indent=2)[:5000] if analysis_json else "No analysis available."

        prompt = f"""{settings.PROMPT_ANALYZER_FOLLOWUP}

DOCUMENT CONTENT:
\"\"\"
{doc_context}
\"\"\"

PREVIOUS ANALYSIS:
{analysis_summary}

USER QUESTION: {question}

Answer the user's question based on the document content and analysis above. Be specific and cite relevant parts of the document."""

        async for chunk in self._stream_gemini_with_retry(
            settings.MODEL_ANALYZER, prompt, temperature=0.4
        ):
            yield f"token: {json.dumps(chunk)}\n"

        yield f"data: {json.dumps({'done': True})}\n"
        logger.info(f"[Analyzer] Follow-up response complete")


analyzer_service = AnalyzerService()
