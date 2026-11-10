"""Load testing runner for the 5 mandatory scenarios."""

import asyncio
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional
import httpx

from relay.chunking import chunk_papers
from relay.config import settings
from relay.metrics import MetricsRecorder
from relay.papers import extract_all_papers
from relay.pipeline import RelayPipeline
from relay.provider_interface import ModelProvider
from relay.schemas import RelayRequest, ResultCSVRow
from relay.standin_provider import StandinProvider


class LoadTestRunner:
    def __init__(
        self,
        provider: ModelProvider,
        source_name: Literal["real", "stand-in"] = "stand-in",
        standin_admin_url: str = "http://127.0.0.1:8001",
    ):
        self.provider = provider
        self.source_name = source_name
        self.standin_admin_url = standin_admin_url
        self.recorder = MetricsRecorder()

        # Load papers and build pipeline
        print("Extracting and indexing papers for load testing...")
        extracted = extract_all_papers()
        chunks = chunk_papers(extracted, chunk_size_tokens=600, overlap_tokens=90)

        paper_pages = {
            p.paper_id: {page.page_number: page.text for page in p.pages}
            for p in extracted
        }
        self.pipeline = RelayPipeline(
            chunks=chunks,
            provider=provider,
            source_name=source_name,
            paper_pages_text=paper_pages,
        )

        self.sample_queries = [
            "What is the frame rate of Mimi in Moshi?",
            "What is Scaled Dot-Product Attention in Vaswani et al.?",
            "How does Dynamic Chunking determine boundaries in H-Net?",
            "How many layers are in the Transformer encoder?",
            "What text model backbone does Moshi use?",
        ]

    async def _configure_standin(self, scenario_name: str):
        """Sets scenario profile on the Stand-in FastAPI service."""
        if self.source_name != "stand-in":
            return
        from standin.app import app as standin_app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=standin_app), base_url="http://standin") as client:
            try:
                await client.post("/admin/reset")
                await client.post(f"/admin/scenario/{scenario_name}")
            except Exception as e:
                print(f"Warning: Could not configure Stand-in admin: {e}")

    async def _send_single_request(
        self, req_id: str, scenario_name: str, query: str
    ) -> ResultCSVRow:
        start_ts = time.time()
        start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts))

        req = RelayRequest(
            question=query,
            request_id=req_id,
            scenario=scenario_name,
            retrieval_setting="setting_b",
        )

        resp = await self.pipeline.execute(req)

        row = ResultCSVRow(
            request_id=resp.request_id,
            scenario=scenario_name,
            source=self.source_name,
            start_time=start_time_iso,
            first_token_ms=resp.first_token_ms,
            total_ms=resp.total_ms,
            status=resp.status,
            retries=resp.retries,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            retrieval_setting="setting_b",
            answer_grade="",  # Empty for load test runs per spec
            queue_wait_ms=resp.stage_timings.queue_wait_ms,
            provider_working_ms=resp.stage_timings.generation_ms,
            retry_wait_ms=resp.stage_timings.retry_wait_ms,
            peak_queue_len=self.pipeline.queue_manager.peak_queue_length,
        )
        self.recorder.record_row(row)
        return row

    # Scenario 1: Steady Load
    async def run_scenario_steady(self, total_requests: int = 15, rate_per_sec: float = 0.5):
        print(f"\n=== Running Scenario 1: Steady Load ({total_requests} requests) ===")
        await self._configure_standin("steady")
        t0 = time.time()
        tasks = []

        for i in range(total_requests):
            req_id = f"steady_{i+1:03d}"
            query = random.choice(self.sample_queries)
            tasks.append(asyncio.create_task(self._send_single_request(req_id, "steady", query)))
            await asyncio.sleep(1.0 / rate_per_sec)

        await asyncio.gather(*tasks)
        duration = time.time() - t0
        print(f"Scenario 1 completed in {duration:.2f}s")

    # Scenario 2: Sudden Burst
    async def run_scenario_burst(self, burst_size: int = 20):
        print(f"\n=== Running Scenario 2: Sudden Burst ({burst_size} concurrent requests) ===")
        await self._configure_standin("burst")
        t0 = time.time()

        tasks = [
            asyncio.create_task(
                self._send_single_request(
                    f"burst_{i+1:03d}",
                    "burst",
                    random.choice(self.sample_queries),
                )
            )
            for i in range(burst_size)
        ]
        await asyncio.gather(*tasks)
        duration = time.time() - t0
        print(f"Scenario 2 completed in {duration:.2f}s")

    # Scenario 3: Sustained Overload
    async def run_scenario_sustained_overload(self, total_requests: int = 35, rate_per_sec: float = 1.5):
        print(f"\n=== Running Scenario 3: Sustained Overload ({total_requests} requests at {rate_per_sec*60:.0f} RPM) ===")
        await self._configure_standin("sustained_overload")
        t0 = time.time()
        tasks = []

        for i in range(total_requests):
            req_id = f"overload_{i+1:03d}"
            query = random.choice(self.sample_queries)
            tasks.append(asyncio.create_task(self._send_single_request(req_id, "sustained_overload", query)))
            await asyncio.sleep(1.0 / rate_per_sec)

        await asyncio.gather(*tasks)
        duration = time.time() - t0
        print(f"Scenario 3 completed in {duration:.2f}s")

    # Scenario 4: Per-minute Limit Halved
    async def run_scenario_rpm_halved(self, total_requests: int = 20, rate_per_sec: float = 0.6):
        print(f"\n=== Running Scenario 4: Per-minute Limit Halved ({total_requests} requests) ===")
        await self._configure_standin("rpm_halved")
        t0 = time.time()
        tasks = []

        for i in range(total_requests):
            req_id = f"halved_{i+1:03d}"
            query = random.choice(self.sample_queries)
            tasks.append(asyncio.create_task(self._send_single_request(req_id, "rpm_halved", query)))
            await asyncio.sleep(1.0 / rate_per_sec)

        await asyncio.gather(*tasks)
        duration = time.time() - t0
        print(f"Scenario 4 completed in {duration:.2f}s")

    # Scenario 5: Daily Budget Exhausted
    async def run_scenario_daily_exhausted(self, total_requests: int = 10):
        print(f"\n=== Running Scenario 5: Daily Budget Exhausted ({total_requests} requests) ===")
        await self._configure_standin("daily_exhausted")
        t0 = time.time()

        tasks = [
            asyncio.create_task(
                self._send_single_request(
                    f"daily_{i+1:03d}",
                    "daily_exhausted",
                    random.choice(self.sample_queries),
                )
            )
            for i in range(total_requests)
        ]
        await asyncio.gather(*tasks)
        duration = time.time() - t0
        print(f"Scenario 5 completed in {duration:.2f}s")


async def main():
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

    provider = StandinProvider()
    runner = LoadTestRunner(provider=provider, source_name="stand-in")

    if scenario == "steady":
        await runner.run_scenario_steady()
    elif scenario == "burst":
        await runner.run_scenario_burst()
    elif scenario == "overload":
        await runner.run_scenario_sustained_overload()
    elif scenario == "rpm_halved":
        await runner.run_scenario_rpm_halved()
    elif scenario == "daily_exhausted":
        await runner.run_scenario_daily_exhausted()
    elif scenario == "all":
        await runner.run_scenario_steady()
        await runner.run_scenario_burst()
        await runner.run_scenario_sustained_overload()
        await runner.run_scenario_rpm_halved()
        await runner.run_scenario_daily_exhausted()
        print("\nAll 5 mandatory scenarios successfully executed! Results logged to results.csv.")


if __name__ == "__main__":
    asyncio.run(main())
