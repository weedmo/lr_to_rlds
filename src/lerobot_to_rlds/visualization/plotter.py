"""Episode plotter for state and action visualization in RLDS datasets."""

from pathlib import Path
from typing import Literal

import numpy as np

from lerobot_to_rlds.visualization.visualizer import DatasetVisualizer


class EpisodePlotter:
    """Matplotlib-based plotter for RLDS episode state and action data."""

    def __init__(self, visualizer: DatasetVisualizer) -> None:
        """Initialize the plotter.

        Args:
            visualizer: DatasetVisualizer instance for the dataset.
        """
        self.visualizer = visualizer

    def _get_episode_data(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract states and actions from an episode.

        Args:
            episode_index: The episode index.

        Returns:
            Tuple of (states, actions) arrays.
        """
        episode = self.visualizer.get_episode(episode_index)
        steps = list(episode["steps"])

        states = []
        actions = []

        for step in steps:
            # Extract state from observation
            obs = step.get("observation", {})
            state = obs.get("state", None)
            if state is not None:
                if hasattr(state, "numpy"):
                    state = state.numpy()
                states.append(state)

            # Extract action
            action = step.get("action", None)
            if action is not None:
                if hasattr(action, "numpy"):
                    action = action.numpy()
                actions.append(action)

        states = np.array(states) if states else np.array([])
        actions = np.array(actions) if actions else np.array([])

        return states, actions

    def plot_state(
        self,
        episode_index: int,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot state over time for an episode.

        Args:
            episode_index: Episode to plot.
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        states, _ = self._get_episode_data(episode_index)

        if states.size == 0:
            print(f"No state data found for episode {episode_index}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        time_steps = np.arange(len(states))

        for dim in range(states.shape[1]):
            ax.plot(time_steps, states[:, dim], label=f"State[{dim}]", alpha=0.8)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("State Value")
        ax.set_title(f"Episode {episode_index} - State over Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_action(
        self,
        episode_index: int,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot action over time for an episode.

        Args:
            episode_index: Episode to plot.
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        _, actions = self._get_episode_data(episode_index)

        if actions.size == 0:
            print(f"No action data found for episode {episode_index}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        time_steps = np.arange(len(actions))

        for dim in range(actions.shape[1]):
            ax.plot(time_steps, actions[:, dim], label=f"Action[{dim}]", alpha=0.8)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Action Value")
        ax.set_title(f"Episode {episode_index} - Action over Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_state_action_overlay(
        self,
        episode_index: int,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot both state and action on the same figure.

        Args:
            episode_index: Episode to plot.
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

        states, actions = self._get_episode_data(episode_index)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        time_steps = np.arange(max(len(states) if states.size else 0, len(actions) if actions.size else 0))

        # Plot states
        if states.size > 0:
            for dim in range(states.shape[1]):
                ax1.plot(time_steps[:len(states)], states[:, dim], label=f"State[{dim}]", alpha=0.8)
            ax1.set_ylabel("State Value")
            ax1.set_title(f"Episode {episode_index} - State")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No state data", ha="center", va="center", transform=ax1.transAxes)

        # Plot actions
        if actions.size > 0:
            for dim in range(actions.shape[1]):
                ax2.plot(time_steps[:len(actions)], actions[:, dim], label=f"Action[{dim}]", alpha=0.8)
            ax2.set_ylabel("Action Value")
            ax2.set_title(f"Episode {episode_index} - Action")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax2.transAxes)

        ax2.set_xlabel("Time Step")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot(
        self,
        episode_index: int,
        plot_type: Literal["state", "action", "both"] = "both",
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
        """Plot episode data.

        Args:
            episode_index: Episode to plot.
            plot_type: Type of plot - 'state', 'action', or 'both'.
            save_path: Optional path to save the figure.
            show: Whether to display the plot.
        """
        if plot_type == "state":
            self.plot_state(episode_index, save_path, show)
        elif plot_type == "action":
            self.plot_action(episode_index, save_path, show)
        else:
            self.plot_state_action_overlay(episode_index, save_path, show)
