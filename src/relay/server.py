"""FastAPI server for The Relay assistant."""

from contextlib import asynccontextmanager
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from relay.chunking import chunk_papers
from relay.config import settings
from relay.groq_provider import GroqProvider
from relay.papers import extract_all_papers
from relay.pipeline import RelayPipeline
from relay.schemas import RelayRequest, RelayResponse
from relay.standin_provider import StandinProvider

pipeline_instance: Optional[RelayPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    print("Initializing The Relay server...")
    extracted = extract_all_papers()
    chunks = chunk_papers(
        extracted,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    paper_pages = {
        p.paper_id: {page.page_number: page.text for page in p.pages}
        for p in extracted
    }

    # Initialize both providers so the server can switch dynamically at runtime
    standin = StandinProvider()
    groq = GroqProvider()

    default_source = "real" if (settings.provider_type == "groq" and settings.groq_api_key) else "stand-in"
    default_provider = groq if default_source == "real" else standin

    pipeline_instance = RelayPipeline(
        chunks=chunks,
        provider=default_provider,
        source_name=default_source,
        paper_pages_text=paper_pages,
        standin_provider=standin,
        groq_provider=groq,
    )
    print(f"Relay ready with {len(chunks)} chunks. Multi-provider switching active (standin, groq).")
    yield


app = FastAPI(title="The Relay API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "the-relay",
        "supported_providers": ["standin", "groq"],
        "default_provider": settings.provider_type,
        "groq_configured": bool(settings.groq_api_key),
        "groq_model": settings.groq_model,
        "standin_model": "standin-calibrated-lpu",
    }


@app.post("/query", response_model=RelayResponse)
@app.post("/ask", response_model=RelayResponse)
async def submit_question(req: RelayRequest):
    """Processes a question with the explicitly requested or default model provider."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return await pipeline_instance.execute(req)


@app.post("/query/stream")
@app.post("/ask/stream")
async def submit_question_stream(req: RelayRequest):
    """Streams question answer tokens and telemetry using Server-Sent Events (SSE)."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    async def event_generator():
        async for delta, final_resp in pipeline_instance.execute_stream(req):
            if delta:
                payload = json.dumps({"delta": delta})
                yield f"data: {payload}\n\n"
            if final_resp:
                payload = json.dumps({"response": final_resp.model_dump()})
                yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
