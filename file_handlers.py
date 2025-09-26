"""Unified file handlers for LightX2V ComfyUI wrapper."""

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from PIL import Image


class FileHandler(ABC):
    """Base class for file handling."""

    @abstractmethod
    def save(self, data: Any, path: str) -> str:
        """Save data to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load data from file."""
        pass


class AudioFileHandler(FileHandler):
    """Handler for audio files."""

    def __init__(self):
        self.supported_formats = [".wav", ".mp3", ".flac", ".m4a"]

    def save(
        self,
        audio_data: Union[Dict, torch.Tensor, np.ndarray, Tuple],
        path: str,
        sample_rate: Optional[int] = None,
    ) -> str:
        """Save audio data to file.

        Args:
            audio_data: Audio data in various formats
            path: Output file path
            sample_rate: Sample rate (required if not in audio_data)

        Returns:
            Path to saved file
        """

        waveform, sr = self._extract_audio_data(audio_data, sample_rate)

        # Ensure waveform is in correct shape
        waveform = self._normalize_waveform(waveform)

        # Always save as WAV format
        ext = os.path.splitext(path)[1].lower()
        if ext != ".wav":
            path = path.rsplit(".", 1)[0] + ".wav"
            logging.info(f"Audio will be saved as WAV format: {path}")

        # Ensure waveform is in int16 format for WAV
        if waveform.dtype != np.int16:
            # Normalize to [-1, 1] range if not already
            if waveform.dtype == np.float32 or waveform.dtype == np.float64:
                # Clip to [-1, 1] to avoid overflow
                waveform = np.clip(waveform, -1.0, 1.0)
                waveform = (waveform * 32767).astype(np.int16)
            else:
                # Assume uint8 or other integer type
                waveform = waveform.astype(np.int16)

        wavfile.write(path, sr, waveform)
        logging.info(f"Audio saved to {path}")
        return path

    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file.

        Returns:
            Tuple of (waveform, sample_rate)
        """
        sample_rate, waveform = wavfile.read(path)
        return waveform, sample_rate

    def _extract_audio_data(self, audio_data: Any, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Extract waveform and sample rate from various audio formats.

        Handles three main sources:
        1. ComfyUI LoadAudio output: {"waveform": tensor, "sample_rate": int}
        2. Tuple format: (waveform, sample_rate)
        3. Raw waveform with separate sample_rate
        """
        if isinstance(audio_data, dict):
            if "waveform" in audio_data and "sample_rate" in audio_data:
                waveform = audio_data["waveform"]
                sr = audio_data["sample_rate"]

                # Handle ComfyUI LoadAudio format specifically
                # ComfyUI returns waveform with shape [batch, channels, samples]
                if isinstance(waveform, torch.Tensor):
                    if waveform.dim() == 3:  # [batch, channels, samples]
                        waveform = waveform[0]  # Take first batch
                    if waveform.dim() == 2 and waveform.shape[0] <= 2:  # [channels, samples]
                        waveform = waveform.transpose(0, 1)  # -> [samples, channels]
                    waveform = waveform.cpu().numpy()
            else:
                raise ValueError("Audio dict must contain 'waveform' and 'sample_rate'")
        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
            waveform, sr = audio_data
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
        elif sample_rate is not None:
            waveform = audio_data
            sr = sample_rate
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
        else:
            raise ValueError("Sample rate must be provided for raw audio data")

        return waveform, sr

    def _normalize_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform shape for saving.

        Ensures waveform is in shape [samples, channels] or [samples] for mono.
        """
        # Already converted to numpy in _extract_audio_data

        # Handle different shapes
        if waveform.ndim == 3:  # Shouldn't happen, but handle it
            waveform = waveform[0]

        if waveform.ndim == 2:
            # Check if it's [channels, samples] format (channels < samples typically)
            if waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
                waveform = waveform.T  # -> [samples, channels]
            # If mono with extra dimension, squeeze it
            if waveform.shape[1] == 1:
                waveform = waveform.squeeze()

        return waveform

    def _save_with_wave(self, path: str, waveform: np.ndarray, sample_rate: int):
        """Save audio using wave module."""
        import wave

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1 if waveform.ndim == 1 else waveform.shape[-1])
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            if waveform.dtype != np.int16:
                waveform = (waveform * 32767).astype(np.int16)

            wav_file.writeframes(waveform.tobytes())

    def _load_with_wave(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio using wave module."""
        import wave

        with wave.open(path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            waveform = np.frombuffer(frames, dtype=np.int16)

            if wav_file.getnchannels() > 1:
                waveform = waveform.reshape(-1, wav_file.getnchannels())

            return waveform, sample_rate


class ImageFileHandler(FileHandler):
    """Handler for image files."""

    def __init__(self):
        self.supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    def save(self, image_data: Union[torch.Tensor, np.ndarray, Image.Image], path: str) -> str:
        """Save image data to file.

        Args:
            image_data: Image data in various formats
            path: Output file path

        Returns:
            Path to saved file
        """
        if isinstance(image_data, torch.Tensor):
            # Convert from tensor [H, W, C] or [C, H, W]
            if image_data.dim() == 4:  # [batch, ...]
                image_data = image_data[0]

            image_np = image_data.cpu().numpy()

            # Handle channel ordering
            if image_np.shape[0] in [1, 3, 4]:  # [C, H, W]
                image_np = np.transpose(image_np, (1, 2, 0))

            # Convert to uint8
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)

            image = Image.fromarray(image_np.squeeze())

        elif isinstance(image_data, np.ndarray):
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            image = Image.fromarray(image_data.squeeze())

        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

        image.save(path)
        logging.info(f"Image saved to {path}")
        return path

    def load(self, path: str) -> Image.Image:
        """Load image from file."""
        return Image.open(path)


class MaskFileHandler(ImageFileHandler):
    """Handler specifically for mask files."""

    def save(self, mask_data: Union[torch.Tensor, np.ndarray], path: str) -> str:
        """Save mask data to file.

        Args:
            mask_data: Mask data (single channel)
            path: Output file path

        Returns:
            Path to saved file
        """
        if isinstance(mask_data, torch.Tensor):
            if mask_data.dim() == 3:  # [batch, H, W]
                mask_data = mask_data[0]
            mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
        elif isinstance(mask_data, np.ndarray):
            mask_np = (mask_data * 255).astype(np.uint8)
        else:
            mask_np = mask_data

        mask_image = Image.fromarray(mask_np)
        mask_image.save(path)
        logging.info(f"Mask saved to {path}")
        return path


class TempFileManager:
    """Manager for temporary files with automatic cleanup."""

    def __init__(self):
        self.temp_files: List[str] = []

    @contextmanager
    def temp_file(self, suffix: str = "", prefix: str = "lightx2v_", delete: bool = True):
        """Context manager for temporary file creation.

        Args:
            suffix: File suffix
            prefix: File prefix
            delete: Whether to delete on exit

        Yields:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        self.temp_files.append(temp_path)

        try:
            yield temp_path
        finally:
            if delete:
                self.cleanup_file(temp_path)

    def create_temp_file(self, suffix: str = "", prefix: str = "lightx2v_") -> str:
        """Create a temporary file that will be tracked for cleanup.

        Args:
            suffix: File suffix
            prefix: File prefix

        Returns:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=False) as tmp:
            temp_path = tmp.name

        self.temp_files.append(temp_path)
        return temp_path

    def cleanup_file(self, path: str):
        """Clean up a specific file."""
        if path in self.temp_files:
            self.temp_files.remove(path)

        if os.path.exists(path):
            try:
                os.unlink(path)
                logging.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logging.warning(f"Failed to clean up {path}: {e}")

    def cleanup_all(self):
        """Clean up all tracked temporary files."""
        for temp_file in self.temp_files[:]:
            self.cleanup_file(temp_file)

        self.temp_files.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_all()


class ComfyUIFileResolver:
    """Resolve file paths for ComfyUI input/output directories."""

    @staticmethod
    def get_input_directory() -> str:
        """Get ComfyUI input directory."""
        try:
            import folder_paths

            return folder_paths.get_input_directory()
        except ImportError:
            # Fallback if not in ComfyUI environment
            return "input"

    @staticmethod
    def resolve_input_path(filename: str) -> str:
        """Resolve a filename to full path in input directory."""
        if os.path.isabs(filename):
            return filename

        input_dir = ComfyUIFileResolver.get_input_directory()
        return os.path.join(input_dir, filename)

    @staticmethod
    def save_to_input(data: Any, filename: str, handler: FileHandler) -> str:
        """Save data to ComfyUI input directory.

        Args:
            data: Data to save
            filename: Target filename
            handler: File handler to use

        Returns:
            Full path to saved file
        """
        input_dir = ComfyUIFileResolver.get_input_directory()
        full_path = os.path.join(input_dir, filename)

        # Create directory if needed
        os.makedirs(input_dir, exist_ok=True)

        return handler.save(data, full_path)
