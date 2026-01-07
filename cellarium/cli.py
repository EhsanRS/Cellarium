"""
Command-line interface for Cellarium.

This module provides the CLI using Typer for starting the Cellarium server
and managing the application.

Usage:
    cellarium serve data.h5ad --port 8050
    cellarium serve data.h5ad --host 0.0.0.0 --port 8080 --debug
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from cellarium import __version__

app = typer.Typer(
    name="cellarium",
    help="Local webapp for exploring AnnData single-cell genomics data.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"Cellarium version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """Cellarium - Single-cell data exploration platform."""
    pass


@app.command()
def serve(
    data_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the .h5ad file containing AnnData object.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host address to bind the server.",
    ),
    port: int = typer.Option(
        8050,
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Port number for the server.",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db",
        "-d",
        help="Path to SQLite database (default: ./cellarium.db).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode with hot reloading.",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't automatically open browser.",
    ),
):
    """
    Start the Cellarium server to explore single-cell data.

    Examples:
        cellarium serve ./my_data.h5ad
        cellarium serve ./my_data.h5ad --port 8080
        cellarium serve ./my_data.h5ad --host 0.0.0.0 --port 8080 --debug
    """
    import os
    from cellarium.config import Config
    from cellarium.db.schema import init_database

    # In debug mode, Flask's reloader spawns a child process.
    # Only show startup messages in the main process (not the reloader).
    is_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"

    # Resolve database path
    if db_path is None:
        db_path = Path.cwd() / "cellarium.db"

    # Display startup banner (only on first run, not reloader)
    if not is_reloader:
        console.print(Panel.fit(
            f"[bold blue]Cellarium[/bold blue] v{__version__}\n"
            f"Single-cell data exploration platform",
            border_style="blue",
        ))

    try:
        # Initialize configuration
        config = Config(
            data_path=data_path,
            db_path=db_path,
            host=host,
            port=port,
            debug=debug,
        )

        if not is_reloader:
            console.print(f"\n[dim]Data file:[/dim] {config.data_path}")
            console.print(f"[dim]Database:[/dim] {config.db_path}")

        # Initialize database
        if not is_reloader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Initializing database...", total=None)
                init_database(config.db_path)
            console.print("[green]Database initialized.[/green]")
        else:
            init_database(config.db_path)

        # Load data and start server
        from cellarium.data.manager import DataManager
        if not is_reloader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading AnnData...", total=None)
                data_manager = DataManager(config)
                progress.update(task, description="Data loaded successfully!")

            console.print(
                f"[green]Loaded:[/green] {data_manager.n_cells:,} cells × "
                f"{data_manager.n_genes:,} genes"
            )
            console.print(
                f"[green]Embeddings:[/green] {', '.join(data_manager.available_embeddings)}"
            )
        else:
            data_manager = DataManager(config)

        # Create and run Dash app
        url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://127.0.0.1:{port}"
        if not is_reloader:
            console.print(f"\n[bold green]Starting server at {url}[/bold green]")
            console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

        from cellarium.app import create_app
        dash_app = create_app(config, data_manager)

        # Auto-open browser after short delay (only on first run, not reloader)
        if not no_browser and not is_reloader:
            import webbrowser
            import threading
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        dash_app.run(
            host=host,
            port=port,
            debug=debug,
        )

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def info(
    data_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the .h5ad file.",
    ),
):
    """
    Display information about an AnnData file without starting the server.
    """
    import anndata as ad

    console.print(f"\n[bold]Loading:[/bold] {data_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Reading file...", total=None)
        adata = ad.read_h5ad(data_path)

    console.print(Panel.fit(
        f"[bold]Cells:[/bold] {adata.n_obs:,}\n"
        f"[bold]Genes:[/bold] {adata.n_vars:,}\n"
        f"[bold]Observations:[/bold] {', '.join(adata.obs.columns[:10])}"
        f"{'...' if len(adata.obs.columns) > 10 else ''}\n"
        f"[bold]Embeddings:[/bold] {', '.join(adata.obsm.keys())}\n"
        f"[bold]Layers:[/bold] {', '.join(adata.layers.keys()) if adata.layers else 'None'}",
        title=f"[bold blue]{data_path.name}[/bold blue]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
