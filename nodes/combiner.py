"""Config combiner nodes.

- V2 ``LightX2VConfigCombinerV2``   : config aggregation + data prep (image/audio/talk_objects),
                                      emits ``PREPARED_CONFIG``.
- V3 ``LightX2VConfigCombinerV3``   : V2 + equal-duration audio padding and background-mask
                                      synthesis for multi-talker setups.
"""

import io
import json
import logging
import os
import subprocess as sp
import wave

import numpy as np
from PIL import Image

from ..config_builder import ConfigBuilder
from ..data_models import (
    InferenceConfig,
    MemoryOptimizationConfig,
    QuantizationConfig,
    TeaCacheConfig,
)
from ..file_handlers import (
    AudioFileHandler,
    ComfyUIFileResolver,
    HTTPFileDownloader,
    ImageFileHandler,
    TempFileManager,
)


class LightX2VConfigCombinerV2:
    """Config combiner that also handles data preparation (image/audio/prompts)."""

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()
        self.http_downloader = HTTPFileDownloader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": (
                    "INFERENCE_CONFIG",
                    {"tooltip": "Basic inference configuration"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Generation prompt"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Negative prompt"},
                ),
            },
            "optional": {
                "teacache_config": (
                    "TEACACHE_CONFIG",
                    {"tooltip": "TeaCache configuration"},
                ),
                "quantization_config": (
                    "QUANT_CONFIG",
                    {"tooltip": "Quantization configuration"},
                ),
                "memory_config": (
                    "MEMORY_CONFIG",
                    {"tooltip": "Memory optimization configuration"},
                ),
                "lora_chain": ("LORA_CHAIN", {"tooltip": "LoRA chain configuration"}),
                "talk_objects_config": ("TALK_OBJECTS_CONFIG", {"tooltip": "Talk objects configuration"}),
                "image": ("IMAGE", {"tooltip": "Input image for i2v or s2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation for s2v or rs2v task"},
                ),
            },
        }

    RETURN_TYPES = ("PREPARED_CONFIG",)
    RETURN_NAMES = ("prepared_config",)
    FUNCTION = "prepare_config"
    CATEGORY = "LightX2V/ConfigV2"

    def prepare_config(
        self,
        inference_config,
        prompt,
        negative_prompt,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        lora_chain=None,
        talk_objects_config=None,
        image=None,
        audio=None,
    ):
        """Combine configurations and prepare data for inference."""

        inf_config = InferenceConfig(**inference_config) if isinstance(inference_config, dict) else inference_config
        tea_config = TeaCacheConfig(**teacache_config) if teacache_config and isinstance(teacache_config, dict) else teacache_config
        quant_config = (
            QuantizationConfig(**quantization_config) if quantization_config and isinstance(quantization_config, dict) else quantization_config
        )
        mem_config = MemoryOptimizationConfig(**memory_config) if memory_config and isinstance(memory_config, dict) else memory_config

        config = self.config_builder.combine_configs(
            inference_config=inf_config,
            teacache_config=tea_config,
            quantization_config=quant_config,
            memory_config=mem_config,
            lora_chain=lora_chain,
            talk_objects_config=talk_objects_config,
        )

        config.prompt = prompt
        config.negative_prompt = negative_prompt

        if config.task in ["i2v", "s2v", "rs2v"] and image is None:
            raise ValueError("i2v or s2v or rs2v task requires input image")

        if config.task in ["i2v", "s2v", "rs2v"] and image is not None:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            temp_path = self.temp_manager.create_temp_file(suffix=".png")
            pil_image.save(temp_path)
            config.image_path = temp_path
            logging.info(f"Image saved to {temp_path}")

        if audio is not None and hasattr(config, "model_cls") and "seko" in config.model_cls:
            temp_path = self.temp_manager.create_temp_file(suffix=".wav")
            self.audio_handler.save(audio, temp_path)
            config.audio_path = temp_path
            logging.info(f"Audio saved to {temp_path}")

        if hasattr(config, "talk_objects") and config.talk_objects:
            talk_objects = config.talk_objects
            processed_talk_objects = []

            for talk_obj in talk_objects:
                processed_obj = {}

                if "audio" in talk_obj:
                    processed_obj["audio"] = talk_obj["audio"]

                if "mask" in talk_obj:
                    processed_obj["mask"] = talk_obj["mask"]

                if "audio" in processed_obj:
                    processed_talk_objects.append(processed_obj)

            for obj in processed_talk_objects:
                if "audio" in obj and obj["audio"]:
                    audio_path = obj["audio"]

                    if self.http_downloader.is_url(audio_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(audio_path, prefix="audio")
                            obj["audio"] = downloaded_path
                            logging.info(f"Downloaded audio from URL: {audio_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download audio from {audio_path}: {e}")
                            continue
                    elif not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                        obj["audio"] = self.resolver.resolve_input_path(audio_path)
                        logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")

                    if not os.path.exists(obj["audio"]):
                        logging.warning(f"Audio file not found: {obj['audio']}")

                if "mask" in obj and obj["mask"]:
                    mask_path = obj["mask"]

                    if self.http_downloader.is_url(mask_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(mask_path, prefix="mask")
                            obj["mask"] = downloaded_path
                            logging.info(f"Downloaded mask from URL: {mask_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download mask from {mask_path}: {e}")
                    elif not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                        obj["mask"] = self.resolver.resolve_input_path(mask_path)
                        logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")

                    if not os.path.exists(obj["mask"]):
                        logging.warning(f"Mask file not found: {obj['mask']}")

            if processed_talk_objects:
                if len(processed_talk_objects) == 1 and not processed_talk_objects[0].get("mask", "").strip():
                    config.audio_path = processed_talk_objects[0]["audio"]
                    logging.info(f"Convert Processed 1 talk object to audio path: {config.audio_path}")
                else:
                    temp_dir = self.temp_manager.create_temp_dir()
                    with open(os.path.join(temp_dir, "config.json"), "w") as f:
                        json.dump({"talk_objects": processed_talk_objects}, f)
                    config.audio_path = temp_dir
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

        logging.info("lightx2v prepared config: " + json.dumps(config, indent=2, ensure_ascii=False))

        return (config,)


class LightX2VConfigCombinerV3:
    """V2 + equal-duration audio padding and background-mask synthesis for multi-talker setups."""

    def __init__(self):
        self.config_builder = ConfigBuilder()
        self.temp_manager = TempFileManager()
        self.image_handler = ImageFileHandler()
        self.audio_handler = AudioFileHandler()
        self.resolver = ComfyUIFileResolver()
        self.http_downloader = HTTPFileDownloader()

    @staticmethod
    def extend_mp3(input_path: str, output_path: str, duration: float) -> bool:
        """Extend or truncate MP3 audio file.

        - If input duration > duration + 0.1, raise an error
        - If input duration is in [duration, duration + 0.1), truncate audio
        - If input duration < duration, extend audio using silence padding
        """
        cmd_probe = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration,sample_rate,bit_rate,channels",
            "-of",
            "json",
            input_path,
        ]

        try:
            output = sp.check_output(cmd_probe, encoding="utf-8", errors="replace")
            data = json.loads(output)
            streams = data.get("streams", [])
            if not streams:
                raise ValueError(f"Failed to get audio stream information: {input_path}")

            stream_info = streams[0]
            input_duration = float(stream_info.get("duration", 0))
            sample_rate = stream_info.get("sample_rate", "44100")
            bit_rate = stream_info.get("bit_rate", "128000")
            channels = stream_info.get("channels", 2)

            if input_duration > duration:
                raise ValueError(f"Input audio duration ({input_duration:.2f}s) exceeds target duration + 0.1s ({duration + 0.1:.2f}s)")
            else:
                pad_duration = duration - input_duration
                cmd = [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-af",
                    f"apad=pad_dur={pad_duration}",
                    "-ar",
                    str(sample_rate),
                    "-b:a",
                    str(bit_rate),
                    "-ac",
                    str(channels),
                    "-c:a",
                    "libmp3lame",
                    "-y",
                    output_path,
                ]

            sp.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace")
            return True

        except sp.CalledProcessError as e:
            if e.stderr:
                logging.error(f"Subprocess execution failed, stderr: {e.stderr}")
            raise
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse audio information: {input_path}")
        except Exception:
            raise

    @staticmethod
    def get_audio_duration(input_path: str) -> float:
        """Get the duration of an audio file in seconds via ffprobe."""
        cmd_probe = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration,sample_rate,bit_rate,channels",
            "-of",
            "json",
            input_path,
        ]
        try:
            output = sp.check_output(cmd_probe, encoding="utf-8", errors="replace")
            data = json.loads(output)
            streams = data.get("streams", [])
            if not streams:
                raise ValueError(f"Failed to get audio stream information: {input_path}")

            stream_info = streams[0]
            return float(stream_info.get("duration", 0))

        except sp.CalledProcessError as e:
            if e.stderr:
                logging.error(f"Subprocess execution failed, stderr: {e.stderr}")
            raise e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse audio information: {input_path}") from e
        except Exception as e:
            raise e

    @staticmethod
    def generate_white_noise(
        duration: float, framerate: int, n_channels: int = 1, rms: float = None, std_dev: float = None, seed: int = None
    ) -> np.ndarray:
        """Generate white noise audio with optional RMS/std-dev normalization."""
        if seed is not None:
            np.random.seed(seed)

        n_samples = int(duration * framerate)

        if n_channels == 1:
            noise = np.random.normal(0, 1, n_samples).astype(np.float32)
        else:
            noise = np.random.normal(0, 1, (n_samples, n_channels)).astype(np.float32)

        if std_dev is not None:
            current_std = np.std(noise)
            if current_std > 0:
                noise = noise * (std_dev / current_std)
        elif rms is not None:
            current_rms = np.sqrt(np.mean(noise**2))
            if current_rms > 0:
                noise = noise * (rms / current_rms)
        return noise

    @staticmethod
    def save_wav_file(audio_data: np.ndarray, output_path, framerate: int, sample_width: int = 2) -> None:
        """Save audio data as WAV file or BytesIO object."""
        if audio_data.ndim == 1:
            n_channels = 1
            audio_data = audio_data.reshape(-1, 1)
        else:
            n_channels = audio_data.shape[1]

        audio_data = np.clip(audio_data, -1.0, 1.0)

        if sample_width == 1:
            audio_int = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        elif sample_width == 2:
            audio_int = (audio_data * 32767).astype(np.int16)
        elif sample_width == 4:
            audio_int = (audio_data * 2147483647).astype(np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels == 1:
            audio_int = audio_int.flatten()
        else:
            audio_int = audio_int.reshape(-1, n_channels)

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(framerate)
            wav_file.writeframes(audio_int.tobytes())

    @staticmethod
    def generate_background_mask(positive_mask_paths):
        """Generate a background mask: white where all positive masks are ~zero, else black."""
        width = None
        height = None
        opened_imgs = []
        for path in positive_mask_paths:
            img = Image.open(path)
            if width is None:
                width = img.width
            elif width != img.width:
                raise ValueError(f"Widths of masks are not the same: {width} != {img.width}")
            if height is None:
                height = img.height
            elif height != img.height:
                raise ValueError(f"Heights of masks are not the same: {height} != {img.height}")
            opened_imgs.append(img)
        img_arrays = []
        for img in opened_imgs:
            img_array = np.array(img)
            if img_array.ndim == 2:
                img_array = img_array[:, :, np.newaxis]
            img_arrays.append(img_array)

        threshold = 1
        zero_masks = []
        for img_array in img_arrays:
            if img_array.shape[-1] == 1:
                zero_mask = img_array[:, :, 0] <= threshold
            else:
                zero_mask = np.all(img_array <= threshold, axis=-1)
            zero_masks.append(zero_mask)

        if zero_masks:
            all_zero_mask = np.logical_and.reduce(zero_masks)
            bg_array = np.where(all_zero_mask, 255, 0).astype(np.uint8)
        else:
            bg_array = np.full((height, width), 255, dtype=np.uint8)

        bg_img = Image.fromarray(bg_array, mode="L")
        img_io = io.BytesIO()
        bg_img.save(img_io, format="JPEG")
        img_io.seek(0)
        for img in opened_imgs:
            img.close()
        return img_io

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_config": (
                    "INFERENCE_CONFIG",
                    {"tooltip": "Basic inference configuration"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Generation prompt"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Negative prompt"},
                ),
            },
            "optional": {
                "teacache_config": (
                    "TEACACHE_CONFIG",
                    {"tooltip": "TeaCache configuration"},
                ),
                "quantization_config": (
                    "QUANT_CONFIG",
                    {"tooltip": "Quantization configuration"},
                ),
                "memory_config": (
                    "MEMORY_CONFIG",
                    {"tooltip": "Memory optimization configuration"},
                ),
                "lora_chain": ("LORA_CHAIN", {"tooltip": "LoRA chain configuration"}),
                "talk_objects_config": ("TALK_OBJECTS_CONFIG", {"tooltip": "Talk objects configuration"}),
                "image": ("IMAGE", {"tooltip": "Input image for i2v or s2v or rs2v task"}),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Input audio for audio-driven generation for s2v or rs2v task"},
                ),
            },
        }

    RETURN_TYPES = ("PREPARED_CONFIG",)
    RETURN_NAMES = ("prepared_config",)
    FUNCTION = "prepare_config"
    CATEGORY = "LightX2V/ConfigV2"

    def prepare_config(
        self,
        inference_config,
        prompt,
        negative_prompt,
        teacache_config=None,
        quantization_config=None,
        memory_config=None,
        lora_chain=None,
        talk_objects_config=None,
        image=None,
        audio=None,
    ):
        """Combine configurations and prepare data for inference."""

        inf_config = InferenceConfig(**inference_config) if isinstance(inference_config, dict) else inference_config
        tea_config = TeaCacheConfig(**teacache_config) if teacache_config and isinstance(teacache_config, dict) else teacache_config
        quant_config = (
            QuantizationConfig(**quantization_config) if quantization_config and isinstance(quantization_config, dict) else quantization_config
        )
        mem_config = MemoryOptimizationConfig(**memory_config) if memory_config and isinstance(memory_config, dict) else memory_config

        config = self.config_builder.combine_configs(
            inference_config=inf_config,
            teacache_config=tea_config,
            quantization_config=quant_config,
            memory_config=mem_config,
            lora_chain=lora_chain,
            talk_objects_config=talk_objects_config,
        )

        config.prompt = prompt
        config.negative_prompt = negative_prompt

        if config.task in ["i2v", "s2v", "rs2v"] and image is None:
            raise ValueError("i2v or s2v or rs2v task requires input image")

        if config.task in ["i2v", "s2v", "rs2v"] and image is not None:
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            temp_path = self.temp_manager.create_temp_file(suffix=".png")
            pil_image.save(temp_path)
            config.image_path = temp_path
            logging.info(f"Image saved to {temp_path}")

        if audio is not None and hasattr(config, "model_cls") and "seko" in config.model_cls:
            temp_path = self.temp_manager.create_temp_file(suffix=".wav")
            self.audio_handler.save(audio, temp_path)
            config.audio_path = temp_path
            logging.info(f"Audio saved to {temp_path}")

        if hasattr(config, "talk_objects") and config.talk_objects:
            talk_objects = config.talk_objects
            src_talk_objects = []

            for talk_obj in talk_objects:
                src_obj = {}

                if "audio" in talk_obj:
                    src_obj["audio"] = talk_obj["audio"]

                if "mask" in talk_obj:
                    src_obj["mask"] = talk_obj["mask"]

                if "audio" in src_obj:
                    src_talk_objects.append(src_obj)

            # Resolve paths / download URLs, and record max source duration.
            max_src_duration = None
            for obj in src_talk_objects:
                if "audio" in obj and obj["audio"]:
                    audio_path = obj["audio"]

                    if self.http_downloader.is_url(audio_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(audio_path, prefix="audio")
                            obj["audio"] = downloaded_path
                            logging.info(f"Downloaded audio from URL: {audio_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download audio from {audio_path}: {e}")
                            continue
                    elif not os.path.isabs(audio_path) and not audio_path.startswith("/tmp"):
                        obj["audio"] = self.resolver.resolve_input_path(audio_path)
                        logging.info(f"Resolved audio path: {audio_path} -> {obj['audio']}")

                    if not os.path.exists(obj["audio"]):
                        logging.warning(f"Audio file not found: {obj['audio']}")
                    duration = self.get_audio_duration(obj["audio"])
                    obj["duration"] = duration
                    if max_src_duration is None or duration > max_src_duration:
                        max_src_duration = duration

                if "mask" in obj and obj["mask"]:
                    mask_path = obj["mask"]

                    if self.http_downloader.is_url(mask_path):
                        try:
                            downloaded_path = self.http_downloader.download_if_url(mask_path, prefix="mask")
                            obj["mask"] = downloaded_path
                            logging.info(f"Downloaded mask from URL: {mask_path} -> {downloaded_path}")
                        except Exception as e:
                            logging.error(f"Failed to download mask from {mask_path}: {e}")
                    elif not os.path.isabs(mask_path) and not mask_path.startswith("/tmp"):
                        obj["mask"] = self.resolver.resolve_input_path(mask_path)
                        logging.info(f"Resolved mask path: {mask_path} -> {obj['mask']}")

                    if not os.path.exists(obj["mask"]):
                        logging.warning(f"Mask file not found: {obj['mask']}")

            if len(src_talk_objects) > 1:
                # Extend each talker's audio to max duration, then synthesize a background track.
                processed_talk_objects = []
                mask_img_paths = []
                extend_count = 0
                for obj in src_talk_objects:
                    dst_obj = {}
                    src_audio_path = obj["audio"]
                    src_audio_duration = obj["duration"]
                    if max_src_duration - src_audio_duration > 0.1:
                        dst_audio_path = self.temp_manager.create_temp_file(suffix=".mp3")
                        self.extend_mp3(src_audio_path, dst_audio_path, max_src_duration)
                        extend_count += 1
                        dst_obj["audio"] = dst_audio_path
                    else:
                        dst_obj["audio"] = src_audio_path
                    src_mask = obj.get("mask", None)
                    if src_mask:
                        dst_obj["mask"] = src_mask
                        mask_img_paths.append(src_mask)
                    processed_talk_objects.append(dst_obj)
                logging.info(f"Extended {extend_count} audio files")

                bg_mask_io = self.generate_background_mask(mask_img_paths)
                bg_mask_path = self.temp_manager.create_temp_file(suffix=".jpg")
                with open(bg_mask_path, "wb") as f:
                    f.write(bg_mask_io.getvalue())
                bg_noise_data = self.generate_white_noise(
                    duration=max_src_duration,
                    framerate=16000,
                    n_channels=1,
                    rms=0.00232,
                    std_dev=0.00232,
                )
                wav_io = io.BytesIO()
                self.save_wav_file(audio_data=bg_noise_data, output_path=wav_io, framerate=16000, sample_width=2)
                bg_audio_path = self.temp_manager.create_temp_file(suffix=".wav")
                with open(bg_audio_path, "wb") as f:
                    f.write(wav_io.getvalue())
                processed_talk_objects.append({"audio": bg_audio_path, "mask": bg_mask_path})
                logging.info(f"Generated background mask and audio: {bg_mask_path}, {bg_audio_path}")
            else:
                processed_talk_objects = src_talk_objects

            if processed_talk_objects:
                if len(processed_talk_objects) == 1 and not processed_talk_objects[0].get("mask", "").strip():
                    config.audio_path = processed_talk_objects[0]["audio"]
                    logging.info(f"Convert Processed 1 talk object to audio path: {config.audio_path}")
                else:
                    temp_dir = self.temp_manager.create_temp_dir()
                    with open(os.path.join(temp_dir, "config.json"), "w") as f:
                        json.dump({"talk_objects": processed_talk_objects}, f)
                    config.audio_path = temp_dir
                    logging.info(f"Processed {len(processed_talk_objects)} talk objects")

        logging.info("lightx2v prepared config: " + json.dumps(config, indent=2, ensure_ascii=False))

        return (config,)
