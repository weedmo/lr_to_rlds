"""Command-line interface for LeRobot to RLDS converter."""

import logging
import sys
from pathlib import Path

import click

from lerobot_to_rlds import __version__
from lerobot_to_rlds.core.types import ConvertMode
from lerobot_to_rlds.pipeline import convert_dataset


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
    default=Path("./output"),
    help="Output directory for RLDS dataset.",
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
    output: Path,
    mode: str,
    format: str,
    resume: bool,
    retry_failed: bool,
) -> None:
    """Convert a LeRobot dataset to RLDS format.

    Runs the full conversion pipeline: discover, spec, convert, validate.
    """
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
        click.echo(click.style("✓ Conversion successful!", fg="green"))
        click.echo(f"  Episodes: {result.episodes_converted}")
        click.echo(f"  Total steps: {result.total_steps}")
        click.echo(f"  Output: {result.output_path}")
    else:
        click.echo("")
        click.echo(click.style("✗ Conversion failed!", fg="red"))
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


if __name__ == "__main__":
    main()
