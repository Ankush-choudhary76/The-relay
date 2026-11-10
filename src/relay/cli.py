"""Command-line interface for running Relay queries and evaluation."""

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from relay.chunking import chunk_papers
from relay.config import settings
from relay.papers import extract_all_papers
from relay.pipeline import RelayPipeline
from relay.schemas import RelayRequest
from relay.standin_provider import StandinProvider

app = typer.Typer(help="The Relay CLI tool.", no_args_is_help=True)
console = Console()


@app.command(name="ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask the research paper library"),
    provider: str = typer.Option("standin", help="Provider to use: 'standin' or 'groq'"),
):
    """Submits a single question through the Relay pipeline."""
    console.print(f"[bold cyan]Running question against research paper library...[/bold cyan]")
    
    extracted = extract_all_papers()
    chunks = chunk_papers(extracted, chunk_size_tokens=600, overlap_tokens=90)
    paper_pages = {
        p.paper_id: {page.page_number: page.text for page in p.pages}
        for p in extracted
    }

    if provider == "groq" and settings.groq_api_key:
        from relay.groq_provider import GroqProvider
        model_provider = GroqProvider()
        source_name = "real"
    else:
        model_provider = StandinProvider()
        source_name = "stand-in"

    pipeline = RelayPipeline(
        chunks=chunks,
        provider=model_provider,
        source_name=source_name,
        paper_pages_text=paper_pages,
    )

    req = RelayRequest(question=question, provider=provider)
    resp = asyncio.run(pipeline.execute(req))

    panel_title = f"Response (status: {resp.status}, source: {resp.source})"
    display_body = resp.answer if resp.status == "ok" else f"[bold red]{resp.refusal_reason or 'No answer returned'}[/bold red]"
    console.print(Panel(display_body, title=panel_title, border_style="green" if resp.status == "ok" else "red"))

    if resp.citations:
        table = Table(title="Verified Citations")
        table.add_column("Paper")
        table.add_column("arXiv ID")
        table.add_column("Page")
        table.add_column("Section")
        for c in resp.citations:
            table.add_row(c.paper_title, f"{c.arxiv_id}{c.version}", str(c.page_number), c.section or "-")
        console.print(table)

    console.print(f"\n[dim]TTFT: {resp.first_token_ms:.1f}ms | Total Latency: {resp.total_ms:.1f}ms | Tokens: {resp.tokens_in} in, {resp.tokens_out} out[/dim]")


@app.command(name="version")
def version():
    """Prints the Relay version and active model info."""
    console.print("[bold green]The Relay v1.0.0[/bold green]")
    console.print(f"Model: {settings.groq_model}")
    console.print(f"Provider: {settings.provider_type}")


if __name__ == "__main__":
    app()

