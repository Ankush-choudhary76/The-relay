"""FastAPI Stand-in model service simulating Groq API behavior."""

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from standin.quota import StandinQuotaTracker
from standin.scenarios import SCENARIO_CONFIGS, StandinConfig
from standin.simulator import StandinSimulator

app = FastAPI(title="The Relay Stand-in Model Service", version="1.0.0")

current_config = StandinConfig()
quota_tracker = StandinQuotaTracker(
    rpm_limit=current_config.rpm_limit,
    tpm_limit=current_config.tpm_limit,
    rpd_limit=current_config.rpd_limit,
    tpd_limit=current_config.tpd_limit,
)
simulator = StandinSimulator(current_config)


class ChatCompletionRequest(BaseModel):
    model: str = "standin-lpu"
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 800
    stream: Optional[bool] = False


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "stand-in-groq-simulator",
        "active_requests": simulator.active_requests,
        "scenario": simulator.config.scenario_name,
    }


@app.get("/admin/status")
def admin_status():
    return {
        "active_requests": simulator.active_requests,
        "scenario": simulator.config.scenario_name,
        "config": simulator.config.model_dump(),
        "daily_requests_used": quota_tracker.daily_requests_used,
        "daily_tokens_used": quota_tracker.daily_tokens_used,
        "forced_daily_exhaustion": quota_tracker.forced_daily_exhaustion,
    }


@app.post("/admin/configure")
def admin_configure(config: StandinConfig):
    global current_config, quota_tracker, simulator
    current_config = config
    quota_tracker.rpm_limit = config.rpm_limit
    quota_tracker.tpm_limit = config.tpm_limit
    quota_tracker.rpd_limit = config.rpd_limit
    quota_tracker.tpd_limit = config.tpd_limit
    simulator.update_config(config)
    return {"status": "configured", "config": config.model_dump()}


@app.post("/admin/scenario/{scenario_name}")
def set_scenario(scenario_name: str):
    if scenario_name not in SCENARIO_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {scenario_name}")
    config = SCENARIO_CONFIGS[scenario_name]
    admin_configure(config)
    if scenario_name == "daily_exhausted":
        quota_tracker.force_exhaust_daily_budget()
    return {"status": "scenario_set", "scenario": scenario_name, "config": config.model_dump()}


@app.post("/admin/reset")
def admin_reset():
    quota_tracker.reset()
    simulator.active_requests = 0
    return {"status": "reset_complete"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # 1. Estimate tokens from prompt
    prompt_text = " ".join(m.get("content", "") for m in req.messages)
    est_prompt_tokens = max(10, len(prompt_text.split()))

    # 2. Check Quota / Rate Limiting
    is_allowed, err_msg, rl_headers = quota_tracker.check_and_consume(est_prompt_tokens)
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": err_msg,
                    "type": "tokens" if "token" in err_msg else "requests",
                    "code": "rate_limit_exceeded",
                }
            },
            headers=rl_headers,
        )

    # 3. Simulated Fault Injection (500, 503)
    if random.random() < simulator.config.error_500_prob:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Simulated Internal Server Error", "code": 500}},
            headers=rl_headers,
        )

    if random.random() < simulator.config.error_503_prob:
        headers = dict(rl_headers)
        headers["retry-after"] = "2"
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Service Temporarily Unavailable", "code": 503}},
            headers=headers,
        )

    # 4. Handle Load Slowdown & Execution
    await simulator.increment_active()
    try:
        ttft_ms = simulator.calculate_ttft()
        synthetic_answer = simulator.generate_synthetic_answer(req.messages)

        # Timeout simulation: sleep indefinitely or past client deadline
        if random.random() < simulator.config.timeout_prob:
            await asyncio.sleep(60.0)

        # 5. Handle Streaming
        if req.stream:
            async def event_generator():
                try:
                    async for chunk in simulator.stream_response(synthetic_answer, ttft_ms):
                        yield chunk
                finally:
                    await simulator.decrement_active()

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        # 6. Handle Non-Streaming
        # Total latency = TTFT + output token synthesis time
        total_tokens = len(synthetic_answer.split())
        gen_duration_ms = ttft_ms + (total_tokens * simulator.config.inter_token_delay_ms)
        await asyncio.sleep(gen_duration_ms / 1000.0)

        res_payload = {
            "id": f"standin-chat-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": synthetic_answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": est_prompt_tokens,
                "completion_tokens": total_tokens,
                "total_tokens": est_prompt_tokens + total_tokens,
            },
        }
        return JSONResponse(content=res_payload, headers=rl_headers)

    finally:
        if not req.stream:
            await simulator.decrement_active()

