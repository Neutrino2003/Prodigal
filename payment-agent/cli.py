"""Interactive CLI for the payment agent."""

import logging

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

from payment_agent import Agent

logging.basicConfig(level=logging.WARNING)

console = Console()
_TITLE = "[bold blue]● Alex[/bold blue]"


def _thinking() -> Live:
    return Live(
        Spinner("dots2", text="  thinking...", style="bold cyan"),
        console=console,
        refresh_per_second=15,
    )


def _show(message: str) -> None:
    with Live(
        Panel("", title=_TITLE, border_style="blue", padding=(0, 1)),
        console=console, refresh_per_second=15,
    ) as live:
        accumulated = ""
        for word in message.split(" "):
            accumulated += ("" if not accumulated else " ") + word
            live.update(Panel(f"[white]{accumulated}[/white]", title=_TITLE, border_style="blue", padding=(0, 1)))


def main() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]Payment Agent[/bold cyan]  [dim]LangGraph + NVIDIA NIM[/dim]\n"
        "[dim]Type [bold]quit[/bold] at any time to exit.[/dim]",
        border_style="cyan", padding=(0, 2),
    ))
    console.print()

    agent = Agent()

    # Opening greeting
    with _thinking():
        result = agent.next("")
    _show(result["message"])

    while not agent.ctx.session_closed:
        console.print()
        user_input = Prompt.ask("[bold green]You[/bold green]", console=console)
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        if not user_input.strip():
            continue

        console.print()
        try:
            with _thinking():
                result = agent.next(user_input)
            _show(result["message"])
        except Exception as exc:
            console.print(f"[red bold]Error:[/red bold] [red]{exc}[/red]")

    if agent.ctx.transaction_id:
        console.print()
        console.print(Panel(
            f"[green]✓  Payment confirmed[/green]\n"
            f"[dim]Transaction:[/dim]  {agent.ctx.transaction_id}",
            border_style="green", padding=(0, 2),
        ))


if __name__ == "__main__":
    main()
