"""Frame viewer for video frames in LeRobot datasets."""

from pathlib import Path

import numpy as np

from lerobot_to_rlds.readers.base import Episode, LeRobotReader


class FrameViewer:
    """Viewer for video frames in LeRobot episodes."""

    def __init__(self, reader: LeRobotReader) -> None:
        """Initialize the frame viewer.

        Args:
            reader: LeRobotReader instance for the dataset.
        """
        self.reader = reader

    def _get_episode(self, episode_index: int) -> Episode:
        """Get episode by index.

        Args:
            episode_index: The episode index.

        Returns:
            Episode object.
        """
        episodes = self.reader.list_episodes()
        for ep_info in episodes:
            if ep_info.episode_index == episode_index:
                return self.reader.read_episode(ep_info)
        raise IndexError(f"Episode index {episode_index} not found")

    def _get_frame(
        self,
        episode: Episode,
        step_idx: int,
        camera: str | None = None,
    ) -> np.ndarray | None:
        """Get a frame from an episode step.

        Args:
            episode: Episode to get frame from.
            step_idx: Step index.
            camera: Camera key (e.g., 'image', 'wrist_image'). If None, uses first available.

        Returns:
            Frame as numpy array (H, W, C) or None if not found.
        """
        if step_idx < 0 or step_idx >= len(episode.steps):
            raise IndexError(f"Step index {step_idx} out of range (0-{len(episode.steps)-1})")

        step = episode.steps[step_idx]

        # Find frame in observation
        if camera is not None:
            # Try exact key and with observation. prefix
            for key in [camera, f"observation.{camera}"]:
                if key in step.observation:
                    frame = step.observation[key]
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                        return frame
            return None

        # Find first image-like observation
        for key, value in step.observation.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                # Likely an image (H, W, C)
                return value

        return None

    def list_cameras(self, episode_index: int) -> list[str]:
        """List available cameras in an episode.

        Args:
            episode_index: Episode to check.

        Returns:
            List of camera keys.
        """
        episode = self._get_episode(episode_index)
        if not episode.steps:
            return []

        cameras = []
        step = episode.steps[0]
        for key, value in step.observation.items():
            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                cameras.append(key)

        return cameras

    def display_frame(
        self,
        episode_index: int,
        step_idx: int,
        camera: str | None = None,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Display a single frame.

        Args:
            episode_index: Episode index.
            step_idx: Step index within the episode.
            camera: Camera to display. If None, uses first available.
            save_path: Optional path to save the image.
            show: Whether to display the image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for viewing. Install with: pip install matplotlib")

        episode = self._get_episode(episode_index)
        frame = self._get_frame(episode, step_idx, camera)

        if frame is None:
            print(f"No frame found for episode {episode_index}, step {step_idx}")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(frame)
        ax.axis("off")
        camera_name = camera or "default"
        ax.set_title(f"Episode {episode_index}, Step {step_idx} ({camera_name})")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved frame to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def display_frame_grid(
        self,
        episode_index: int,
        step_indices: list[int],
        camera: str | None = None,
        cols: int = 4,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Display multiple frames in a grid.

        Args:
            episode_index: Episode index.
            step_indices: List of step indices to display.
            camera: Camera to display. If None, uses first available.
            cols: Number of columns in the grid.
            save_path: Optional path to save the image.
            show: Whether to display the image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for viewing. Install with: pip install matplotlib")

        episode = self._get_episode(episode_index)

        frames = []
        valid_indices = []
        for idx in step_indices:
            frame = self._get_frame(episode, idx, camera)
            if frame is not None:
                frames.append(frame)
                valid_indices.append(idx)

        if not frames:
            print(f"No frames found for episode {episode_index}")
            return

        rows = (len(frames) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

        # Flatten axes for easier indexing
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i, (frame, idx) in enumerate(zip(frames, valid_indices)):
            row, col = i // cols, i % cols
            axes[row][col].imshow(frame)
            axes[row][col].axis("off")
            axes[row][col].set_title(f"Step {idx}")

        # Hide empty subplots
        for i in range(len(frames), rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].axis("off")

        camera_name = camera or "default"
        fig.suptitle(f"Episode {episode_index} - {camera_name}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved grid to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_frames_as_images(
        self,
        episode_index: int,
        output_dir: Path,
        camera: str | None = None,
        step_range: tuple[int, int] | None = None,
    ) -> list[Path]:
        """Save episode frames as PNG images.

        Args:
            episode_index: Episode to export.
            output_dir: Directory to save images.
            camera: Camera to export. If None, uses first available.
            step_range: Optional (start, end) range of steps. If None, exports all.

        Returns:
            List of saved file paths.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pillow is required for saving. Install with: pip install pillow")

        episode = self._get_episode(episode_index)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if step_range:
            start, end = step_range
            indices = range(start, min(end, len(episode.steps)))
        else:
            indices = range(len(episode.steps))

        saved_paths = []
        for idx in indices:
            frame = self._get_frame(episode, idx, camera)
            if frame is not None:
                # Ensure uint8 format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                img = Image.fromarray(frame)
                camera_name = camera or "cam"
                path = output_dir / f"ep{episode_index:04d}_step{idx:06d}_{camera_name}.png"
                img.save(path)
                saved_paths.append(path)

        print(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths
