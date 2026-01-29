"""Custom exceptions for LeRobot to RLDS converter."""


class LeRobotToRLDSError(Exception):
    """Base exception for all converter errors."""

    pass


class VersionDetectionError(LeRobotToRLDSError):
    """Failed to detect LeRobot version."""

    pass


class ValidationError(LeRobotToRLDSError):
    """Validation failed."""

    pass


class ConversionError(LeRobotToRLDSError):
    """Conversion failed."""

    pass


class EpisodeProcessingError(ConversionError):
    """Failed to process a specific episode."""

    def __init__(self, episode_id: str, message: str):
        self.episode_id = episode_id
        super().__init__(f"Episode {episode_id}: {message}")


class VideoDecodeError(EpisodeProcessingError):
    """Failed to decode video file."""

    pass


class SchemaError(LeRobotToRLDSError):
    """Schema mismatch or invalid schema."""

    pass


class CheckpointError(LeRobotToRLDSError):
    """Error with checkpoint file."""

    pass
