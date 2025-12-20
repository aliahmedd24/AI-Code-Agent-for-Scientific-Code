#!/usr/bin/env python3
"""
Scientific Agent System - Main Entry Point

A fully autonomous LLM-driven agentic system for analyzing scientific papers
and generating executable code from GitHub repositories.

Usage:
    # Run the web server
    python main.py serve
    
    # Run analysis from command line
    python main.py analyze --paper <url> --repo <url>
    
    # Interactive mode
    python main.py interactive
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🔬 Scientific Agent System                                  ║
    ║                                                               ║
    ║   LLM-Powered Pipeline for Scientific Paper Analysis          ║
    ║   and Automatic Code Generation                               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


async def run_analysis(
    paper_url: str,
    repo_url: str,
    gemini_api_key: str,
    github_token: Optional[str] = None,
    output_dir: str = "./outputs",
    use_docker: bool = True,
    verbose: bool = True
):
    """Run the complete analysis pipeline."""
    from core.orchestrator import PipelineOrchestrator, PipelineEvent
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        gemini_api_key=gemini_api_key,
        github_token=github_token,
        output_dir=output_dir,
        use_docker=use_docker
    )
    
    # Progress callback
    if verbose:
        def on_event(event: PipelineEvent):
            icon = "✓" if not event.is_error else "✗"
            color = "green" if not event.is_error else "red"
            console.print(f"[{color}]{icon}[/{color}] [{event.stage.value}] {event.message}")
        
        orchestrator.add_callback(on_event)
    
    # Run pipeline
    console.print("\n[bold cyan]Starting analysis...[/bold cyan]\n")
    
    result = await orchestrator.run_pipeline(
        paper_url=paper_url,
        repo_url=repo_url,
        auto_fix_errors=True
    )
    
    return result


def display_results(result):
    """Display analysis results in a formatted way."""
    console.print("\n")
    
    if result.success:
        console.print(Panel("[bold green]✓ Analysis Completed Successfully![/bold green]", style="green"))
    else:
        console.print(Panel(f"[bold red]✗ Analysis Failed: {result.error}[/bold red]", style="red"))
    
    # Summary table
    table = Table(title="Analysis Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Paper", result.paper_info.get("title", "N/A")[:50])
    table.add_row("Repository", result.repo_info.get("name", "N/A"))
    table.add_row("Concept Mappings", str(len(result.concept_mappings)))
    table.add_row("Generated Files", str(len(result.generated_code)))
    table.add_row("Visualizations", str(len(result.visualizations)))
    table.add_row("Total Time", f"{result.total_time:.1f}s")
    
    if result.report_path:
        table.add_row("Report", result.report_path)
    
    console.print(table)
    
    # Show concept mappings
    if result.concept_mappings:
        console.print("\n[bold]Concept Mappings:[/bold]")
        for mapping in result.concept_mappings[:5]:
            concept = mapping.get("concept_name", "Unknown")
            code = mapping.get("code_element", "N/A")
            confidence = mapping.get("confidence", 0)
            console.print(f"  • {concept} → [cyan]{code}[/cyan] ({confidence*100:.0f}%)")
    
    # Show generated code files
    if result.generated_code:
        console.print("\n[bold]Generated Code Files:[/bold]")
        for code in result.generated_code:
            console.print(f"  📄 {code['filename']} - {code['purpose'][:50]}")


def serve_command(args):
    """Start the web server."""
    from api.server import run_server
    
    print_banner()
    console.print("[bold]Starting web server...[/bold]\n")
    console.print(f"[dim]Binding to: {args.host}:{args.port}[/dim]")
    console.print(f"\n[bold cyan]Open http://localhost:{args.port} in your browser[/bold cyan]")
    console.print("[dim](If running on a remote server, use the server's IP address)[/dim]\n")
    
    run_server(host=args.host, port=args.port)


def analyze_command(args):
    """Run analysis from command line."""
    print_banner()
    
    # Get API key
    gemini_api_key = args.gemini_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        console.print("[red]Error: Gemini API key required. Set GEMINI_API_KEY or use --gemini-key[/red]")
        sys.exit(1)
    
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    
    # Run analysis
    result = asyncio.run(run_analysis(
        paper_url=args.paper,
        repo_url=args.repo,
        gemini_api_key=gemini_api_key,
        github_token=github_token,
        output_dir=args.output,
        use_docker=not args.no_docker,
        verbose=not args.quiet
    ))
    
    # Display results
    display_results(result)
    
    if result.report_path:
        console.print(f"\n[bold]📊 Full report available at:[/bold] {result.report_path}")


def interactive_command(args):
    """Run in interactive mode."""
    print_banner()
    
    console.print("[bold]Interactive Mode[/bold]\n")
    console.print("Enter the required information below:\n")
    
    # Get inputs
    gemini_key = os.getenv("GEMINI_API_KEY") or console.input("[cyan]Gemini API Key: [/cyan]")
    github_token = os.getenv("GITHUB_TOKEN") or console.input("[cyan]GitHub Token (optional, press Enter to skip): [/cyan]")
    
    paper_url = console.input("[cyan]Paper URL (arXiv or PDF): [/cyan]")
    repo_url = console.input("[cyan]GitHub Repository URL: [/cyan]")
    
    use_docker = console.input("[cyan]Use Docker sandbox? (Y/n): [/cyan]").lower() != 'n'
    
    if not paper_url or not repo_url:
        console.print("[red]Error: Both paper and repo URLs are required[/red]")
        return
    
    # Run analysis
    result = asyncio.run(run_analysis(
        paper_url=paper_url,
        repo_url=repo_url,
        gemini_api_key=gemini_key,
        github_token=github_token if github_token else None,
        use_docker=use_docker
    ))
    
    display_results(result)


def demo_command(args):
    """Run a demo with sample inputs."""
    print_banner()
    
    console.print("[bold yellow]Running Demo Mode[/bold yellow]\n")
    console.print("This demo will analyze a sample paper and repository.\n")
    
    # Sample inputs (using well-known open source projects)
    sample_paper = "https://arxiv.org/abs/1706.03762"  # Attention Is All You Need
    sample_repo = "https://github.com/pytorch/pytorch"
    
    console.print(f"[dim]Paper: {sample_paper}[/dim]")
    console.print(f"[dim]Repo: {sample_repo}[/dim]\n")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        console.print("[red]Error: Set GEMINI_API_KEY environment variable[/red]")
        return
    
    result = asyncio.run(run_analysis(
        paper_url=sample_paper,
        repo_url=sample_repo,
        gemini_api_key=gemini_key,
        use_docker=False  # Simpler for demo
    ))
    
    display_results(result)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scientific Agent System - LLM-powered paper analysis and code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the web server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis from command line")
    analyze_parser.add_argument("--paper", "-p", required=True, help="Paper URL (arXiv or PDF)")
    analyze_parser.add_argument("--repo", "-r", required=True, help="GitHub repository URL")
    analyze_parser.add_argument("--gemini-key", help="Gemini API key (or set GEMINI_API_KEY)")
    analyze_parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    analyze_parser.add_argument("--output", "-o", default="./outputs", help="Output directory")
    analyze_parser.add_argument("--no-docker", action="store_true", help="Don't use Docker sandbox")
    analyze_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample inputs")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        serve_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "interactive":
        interactive_command(args)
    elif args.command == "demo":
        demo_command(args)
    else:
        print_banner()
        parser.print_help()
        console.print("\n[dim]Examples:[/dim]")
        console.print("  python main.py serve                    # Start web interface")
        console.print("  python main.py analyze -p <paper> -r <repo>  # CLI analysis")
        console.print("  python main.py interactive              # Interactive mode")


if __name__ == "__main__":
    main()
