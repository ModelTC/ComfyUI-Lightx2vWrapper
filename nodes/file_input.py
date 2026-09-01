"""Validated file-path inputs for native LightX2V media runners."""

from pathlib import Path

import folder_paths


def _input_video_files():
    input_dir = folder_paths.get_input_directory()
    files, _ = folder_paths.recursive_search(input_dir)
    return sorted(folder_paths.filter_files_content_types(files, ["video"]))


def resolve_input_video_path(filename) -> Path:
    """Resolve a ComfyUI input filename without allowing directory escape."""

    raw = str(filename or "").strip()
    if not raw:
        raise ValueError("video is required")

    input_dir = Path(folder_paths.get_input_directory()).resolve()
    candidate = Path(folder_paths.get_annotated_filepath(raw)).resolve()
    try:
        candidate.relative_to(input_dir)
    except ValueError as exc:
        raise ValueError(f"Expected a video under ComfyUI input, got: {filename}") from exc

    if not candidate.is_file():
        raise FileNotFoundError(f"Input video does not exist: {candidate}")
    if not folder_paths.filter_files_content_types([candidate.name], ["video"]):
        raise ValueError(f"Input file is not recognized as video: {candidate}")
    return candidate


def probe_video_file(video_path: Path):
    """Read only video metadata and the first frame dimensions via decord."""

    from decord import VideoReader

    reader = VideoReader(str(video_path))
    if len(reader) < 1:
        raise ValueError(f"Input video contains no frames: {video_path}")
    first_frame = reader[0]
    height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    fps = float(reader.get_avg_fps() or 0.0)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid input video dimensions: {width}x{height}")
    return width, height, fps


class LightX2VInputVideoPath:
    """Upload/select a video under ComfyUI input and expose its absolute path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    _input_video_files(),
                    {
                        "video_upload": True,
                        "tooltip": "Upload or select a video under ComfyUI input. The absolute path is resolved only while executing.",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "resolve"
    CATEGORY = "LightX2V/Input"

    def resolve(self, video):
        return (str(resolve_input_video_path(video)),)

    @classmethod
    def IS_CHANGED(cls, video):
        path = resolve_input_video_path(video)
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        try:
            resolve_input_video_path(video)
        except (OSError, ValueError) as exc:
            return str(exc)
        return True
