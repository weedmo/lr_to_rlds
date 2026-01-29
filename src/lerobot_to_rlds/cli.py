"""Command-line interface for LeRobot to RLDS converter."""

import logging
import sys
from pathlib import Path

import click

from lerobot_to_rlds import __version__
from lerobot_to_rlds.core.types import ConvertMode
from lerobot_to_rlds.pipeline import convert_dataset
from lerobot_to_rlds.utils.naming import get_output_path, DEFAULT_OUTPUT_BASE


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """LeRobot to RLDS Converter.

    Convert LeRobot datasets (v2.1/v3.0) to RLDS (TFDS) format.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for inventory files.",
)
def discover(dataset_path: Path, output: Path) -> None:
    """Discover and inventory a LeRobot dataset.

    Scans the dataset, detects version, and generates inventory.json
    and episode_index.csv.
    """
    from lerobot_to_rlds.readers import get_reader

    click.echo(f"Discovering dataset: {dataset_path}")

    try:
        reader = get_reader(dataset_path)
        click.echo(f"LeRobot version: {reader.version.value}")
        click.echo(f"Dataset name: {reader.dataset_name}")
        click.echo(f"Episodes: {reader.episode_count}")
        click.echo(f"Total steps: {reader.total_steps}")

        # List episodes
        episodes = reader.list_episodes()
        click.echo(f"\nEpisode summary:")
        for ep in episodes[:5]:
            click.echo(f"  {ep.episode_id}: {ep.length} steps, task='{ep.task[:50]}...' " if len(ep.task) > 50 else f"  {ep.episode_id}: {ep.length} steps, task='{ep.task}'")
        if len(episodes) > 5:
            click.echo(f"  ... and {len(episodes) - 5} more")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for RLDS dataset. Defaults to data/<task_name>/.",
)
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Custom name for the output folder (used with default output path).",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["safe", "parallel-half", "parallel-max"]),
    default="safe",
    help="Execution mode for conversion.",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["oxe", "legacy"]),
    default="oxe",
    help="Output format. 'oxe' for OXE/OpenVLA compatible (default), 'legacy' for old format.",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from previous checkpoint.",
)
@click.option(
    "--retry-failed/--no-retry-failed",
    default=False,
    help="Retry previously failed episodes.",
)
def convert(
    dataset_path: Path,
    output: Path | None,
    name: str | None,
    mode: str,
    format: str,
    resume: bool,
    retry_failed: bool,
) -> None:
    """Convert a LeRobot dataset to RLDS format.

    Runs the full conversion pipeline: discover, spec, convert, validate.

    If --output is not specified, outputs to data/<folder_name>/ by default.
    Use --name to customize the output folder name.
    """
    # Determine output path
    if output is None:
        output = get_output_path(dataset_path, output_name=name)

    convert_mode = ConvertMode(mode)
    click.echo(f"Converting dataset: {dataset_path}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Mode: {convert_mode.value}")
    click.echo(f"Format: {format}")
    click.echo(f"Resume: {resume}")
    click.echo("")

    result = convert_dataset(
        dataset_path=dataset_path,
        output_dir=output,
        mode=convert_mode,
        resume=resume,
        retry_failed=retry_failed,
        output_format=format,
    )

    if result.success:
        click.echo("")
        click.echo(click.style("Conversion successful!", fg="green"))
        click.echo(f"  Episodes: {result.episodes_converted}")
        click.echo(f"  Total steps: {result.total_steps}")
        click.echo(f"  Output: {result.output_path}")
    else:
        click.echo("")
        click.echo(click.style("Conversion failed!", fg="red"))
        for error in result.errors:
            click.echo(f"  - {error}", err=True)
        raise SystemExit(1)


@main.command()
@click.argument("rlds_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--original", "-o",
    type=click.Path(exists=True, path_type=Path),
    help="Original LeRobot dataset for comparison.",
)
def validate(rlds_path: Path, original: Path | None) -> None:
    """Validate a converted RLDS dataset.

    Checks data integrity, schema compliance, and optionally
    compares against the original dataset.
    """
    click.echo(f"Validating RLDS dataset: {rlds_path}")
    if original:
        click.echo(f"Comparing with original: {original}")
    # TODO: Implement validation
    click.echo("Validation not yet implemented.")


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory.",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["safe", "parallel-half", "parallel-max"]),
    default="safe",
    help="Execution mode for conversion.",
)
def run(dataset_path: Path, output: Path, mode: str) -> None:
    """Run full conversion pipeline.

    Equivalent to: discover -> spec -> convert -> validate -> publish
    """
    convert_mode = ConvertMode(mode)
    click.echo(f"Running full pipeline on: {dataset_path}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Mode: {convert_mode.value}")
    # TODO: Implement full pipeline
    click.echo("Full pipeline not yet implemented.")


@main.command("list-datasets")
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_OUTPUT_BASE,
    help=f"Data directory to scan. Defaults to '{DEFAULT_OUTPUT_BASE}'.",
)
def list_datasets(data_dir: Path) -> None:
    """List RLDS datasets in the data directory."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        click.echo(f"Data directory does not exist: {data_dir}")
        return

    # Find subdirectories that look like RLDS datasets
    datasets = []
    for path in sorted(data_dir.iterdir()):
        if path.is_dir():
            # Check if it looks like an RLDS dataset (has dataset_info.json)
            if (path / "dataset_info.json").exists():
                datasets.append((path.name, "RLDS"))
            elif any(path.glob("*/dataset_info.json")):
                # Nested structure
                datasets.append((path.name, "RLDS"))

    if not datasets:
        click.echo(f"No RLDS datasets found in {data_dir}")
        return

    click.echo(f"\nRLDS Datasets in {data_dir}:")
    click.echo("-" * 50)
    click.echo(f"{'Name':<35} {'Type':<15}")
    click.echo("-" * 50)
    for name, dtype in datasets:
        click.echo(f"{name:<35} {dtype:<15}")
    click.echo("-" * 50)
    click.echo(f"Total: {len(datasets)} dataset(s)")


# Visualization command group
@main.group()
def visualize() -> None:
    """Visualize RLDS datasets."""
    pass


@visualize.command("list")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
@click.option(
    "--max-episodes", "-m",
    type=int,
    default=100,
    help="Maximum number of episodes to display.",
)
def visualize_list(dataset_path: Path, name: str | None, max_episodes: int) -> None:
    """List episodes in an RLDS dataset."""
    from lerobot_to_rlds.visualization import DatasetVisualizer

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        viz.print_episodes_table(max_episodes=max_episodes)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@visualize.command("info")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
def visualize_info(dataset_path: Path, name: str | None) -> None:
    """Show detailed RLDS dataset information."""
    from lerobot_to_rlds.visualization import DatasetVisualizer

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        viz.print_dataset_info()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@visualize.command("plot")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
@click.option(
    "--episode", "-e",
    type=int,
    default=0,
    help="Episode index to plot.",
)
@click.option(
    "--type", "-t",
    "plot_type",
    type=click.Choice(["state", "action", "both"]),
    default="both",
    help="Type of plot.",
)
@click.option(
    "--save", "-s",
    type=click.Path(path_type=Path),
    default=None,
    help="Save plot to file instead of displaying.",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Don't display the plot (use with --save).",
)
def visualize_plot(
    dataset_path: Path,
    name: str | None,
    episode: int,
    plot_type: str,
    save: Path | None,
    no_show: bool,
) -> None:
    """Plot state and/or action for an episode."""
    from lerobot_to_rlds.visualization import DatasetVisualizer, EpisodePlotter

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        plotter = EpisodePlotter(viz)
        plotter.plot(episode, plot_type=plot_type, save_path=save, show=not no_show)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install matplotlib with: pip install matplotlib", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@visualize.command("frames")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
@click.option(
    "--episode", "-e",
    type=int,
    default=0,
    help="Episode index.",
)
@click.option(
    "--step", "-s",
    type=int,
    default=None,
    help="Single step index to display.",
)
@click.option(
    "--steps",
    type=str,
    default=None,
    help="Comma-separated step indices for grid view (e.g., '0,10,20,30').",
)
@click.option(
    "--camera", "-c",
    type=str,
    default="image",
    help="Camera key to display (default: 'image').",
)
@click.option(
    "--save", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save frame(s) to file.",
)
@click.option(
    "--no-show",
    is_flag=True,
    help="Don't display (use with --save).",
)
def visualize_frames(
    dataset_path: Path,
    name: str | None,
    episode: int,
    step: int | None,
    steps: str | None,
    camera: str,
    save: Path | None,
    no_show: bool,
) -> None:
    """View image frames from an RLDS episode."""
    from lerobot_to_rlds.visualization import DatasetVisualizer, FrameViewer

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        viewer = FrameViewer(viz)

        if steps:
            # Grid view
            step_indices = [int(s.strip()) for s in steps.split(",")]
            viewer.display_frame_grid(
                episode, step_indices, camera=camera,
                save_path=save, show=not no_show
            )
        else:
            # Single frame view
            step_idx = step if step is not None else 0
            viewer.display_frame(
                episode, step_idx, camera=camera,
                save_path=save, show=not no_show
            )
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install matplotlib with: pip install matplotlib", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@visualize.command("cameras")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
@click.option(
    "--episode", "-e",
    type=int,
    default=0,
    help="Episode index to check.",
)
def visualize_cameras(dataset_path: Path, name: str | None, episode: int) -> None:
    """List available cameras in an RLDS episode."""
    from lerobot_to_rlds.visualization import DatasetVisualizer, FrameViewer

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        viewer = FrameViewer(viz)
        cameras = viewer.list_cameras(episode)

        if cameras:
            click.echo(f"Cameras in episode {episode}:")
            for cam in cameras:
                click.echo(f"  - {cam}")
        else:
            click.echo(f"No cameras found in episode {episode}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@visualize.command("export-frames")
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--name", "-n",
    type=str,
    default=None,
    help="Dataset name (auto-detected if not provided).",
)
@click.option(
    "--episode", "-e",
    type=int,
    default=0,
    help="Episode index.",
)
@click.option(
    "--camera", "-c",
    type=str,
    default="image",
    help="Camera to export (default: 'image').",
)
@click.option(
    "--start",
    type=int,
    default=None,
    help="Start step index.",
)
@click.option(
    "--end",
    type=int,
    default=None,
    help="End step index.",
)
def visualize_export_frames(
    dataset_path: Path,
    output_dir: Path,
    name: str | None,
    episode: int,
    camera: str,
    start: int | None,
    end: int | None,
) -> None:
    """Export image frames from RLDS episode as PNG images."""
    from lerobot_to_rlds.visualization import DatasetVisualizer, FrameViewer

    try:
        viz = DatasetVisualizer(dataset_path, dataset_name=name)
        viewer = FrameViewer(viz)

        step_range = None
        if start is not None or end is not None:
            s = start or 0
            e = end or 999999
            step_range = (s, e)

        paths = viewer.save_frames_as_images(
            episode, output_dir, camera=camera, step_range=step_range
        )
        click.echo(f"Exported {len(paths)} frames to {output_dir}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
