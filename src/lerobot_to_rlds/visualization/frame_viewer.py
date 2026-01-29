"""Frame viewer for images in RLDS datasets."""

from pathlib import Path

import numpy as np

from lerobot_to_rlds.visualization.visualizer import DatasetVisualizer


class FrameViewer:
    """Viewer for image frames in RLDS episodes."""

    def __init__(self, visualizer: DatasetVisualizer) -> None:
        """Initialize the frame viewer.

        Args:
            visualizer: DatasetVisualizer instance for the dataset.
        """
        self.visualizer = visualizer

    def _get_frame(
        self,
        episode_index: int,
        step_idx: int,
        camera: str = "image",
    ) -> np.ndarray | None:
        """Get a frame from an episode step.

        Args:
            episode_index: Episode index.
            step_idx: Step index.
            camera: Camera key (e.g., 'image', 'wrist_image').

        Returns:
            Frame as numpy array (H, W, C) or None if not found.
        """
        episode = self.visualizer.get_episode(episode_index)
        steps = list(episode["steps"])

        if step_idx < 0 or step_idx >= len(steps):
            raise IndexError(f"Step index {step_idx} out of range (0-{len(steps)-1})")

        step = steps[step_idx]
        obs = step.get("observation", {})

        frame = obs.get(camera, None)
        if frame is None:
            return None

        if hasattr(frame, "numpy"):
            frame = frame.numpy()

        return frame

    def list_cameras(self, episode_index: int) -> list[str]:
        """List available cameras in an episode.

        Args:
            episode_index: Episode to check.

        Returns:
            List of camera keys.
        """
        episode = self.visualizer.get_episode(episode_index)
        steps = list(episode["steps"])

        if not steps:
            return []

        cameras = []
        obs = steps[0].get("observation", {})

        # Known image keys in OXE schema
        for key in ["image", "wrist_image"]:
            if key in obs:
                cameras.append(key)

        # Also check for any other image-like observations
        for key, value in obs.items():
            if key not in cameras:
                if hasattr(value, "numpy"):
                    value = value.numpy()
                if isinstance(value, np.ndarray) and len(value.shape) == 3 and value.shape[-1] in [1, 3, 4]:
                    cameras.append(key)

        return cameras

    def display_frame(
        self,
        episode_index: int,
        step_idx: int,
        camera: str = "image",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Display a single frame.

        Args:
            episode_index: Episode index.
            step_idx: Step index within the episode.
            camera: Camera to display.
            save_path: Optional path to save the image.
            show: Whether to display the image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for viewing. Install with: pip install matplotlib")

        frame = self._get_frame(episode_index, step_idx, camera)

        if frame is None:
            print(f"No frame found for episode {episode_index}, step {step_idx}, camera '{camera}'")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(f"Episode {episode_index}, Step {step_idx} ({camera})")

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
        camera: str = "image",
        cols: int = 4,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Display multiple frames in a grid.

        Args:
            episode_index: Episode index.
            step_indices: List of step indices to display.
            camera: Camera to display.
            cols: Number of columns in the grid.
            save_path: Optional path to save the image.
            show: Whether to display the image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for viewing. Install with: pip install matplotlib")

        frames = []
        valid_indices = []
        for idx in step_indices:
            try:
                frame = self._get_frame(episode_index, idx, camera)
                if frame is not None:
                    frames.append(frame)
                    valid_indices.append(idx)
            except IndexError:
                continue

        if not frames:
            print(f"No frames found for episode {episode_index}, camera '{camera}'")
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

        fig.suptitle(f"Episode {episode_index} - {camera}", fontsize=14)
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
        camera: str = "image",
        step_range: tuple[int, int] | None = None,
    ) -> list[Path]:
        """Save episode frames as PNG images.

        Args:
            episode_index: Episode to export.
            output_dir: Directory to save images.
            camera: Camera to export.
            step_range: Optional (start, end) range of steps. If None, exports all.

        Returns:
            List of saved file paths.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("pillow is required for saving. Install with: pip install pillow")

        episode = self.visualizer.get_episode(episode_index)
        steps = list(episode["steps"])

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if step_range:
            start, end = step_range
            indices = range(start, min(end, len(steps)))
        else:
            indices = range(len(steps))

        saved_paths = []
        for idx in indices:
            frame = self._get_frame(episode_index, idx, camera)
            if frame is not None:
                # Ensure uint8 format
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)

                img = Image.fromarray(frame)
                path = output_dir / f"ep{episode_index:04d}_step{idx:06d}_{camera}.png"
                img.save(path)
                saved_paths.append(path)

        print(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths
