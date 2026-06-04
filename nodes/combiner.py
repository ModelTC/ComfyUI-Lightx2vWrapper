"""Config combiner nodes.

- V2 ``LightX2VConfigCombinerV2`` : config aggregation + data prep (image/audio/talk_objects),
                                    emits ``PREPARED_CONFIG``.
- V3 ``LightX2VConfigCombinerV3`` : V2 + equal-duration audio padding and background-mask
                                    synthesis for multi-talker setups (used when the user's
                                    per-speaker audios differ in length and must be aligned).

V2 and V3 share INPUT_TYPES and most of ``prepare_config``; the shared scaffolding lives
in the private ``_BaseConfigCombiner`` below. V3 only overrides the multi-talker branch
to add padding + background track synthesis.
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


class _BaseConfigCombiner:
    """Shared scaffolding for V2 / V3.

    Subclasses must implement ``_process_talk_objects(src_objects, max_duration)``
    returning the final ``processed_talk_objects`` list (V2 passes through;
    V3 pads to equal length and appends a background talker).
    """

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

    # --- pipeline ---------------------------------------------------------

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
        config = self._build_base_config(
            inference_config,
            prompt,
            negative_prompt,
            teacache_config,
            quantization_config,
            memory_config,
            lora_chain,
            talk_objects_config,
        )
        self._save_image_if_needed(config, image)
        self._save_single_audio_if_needed(config, audio)
        self._handle_talk_objects(config)

        logging.info("lightx2v prepared config: " + json.dumps(config, indent=2, ensure_ascii=False))
        return (config,)

    # --- shared helpers ---------------------------------------------------

    def _build_base_config(
        self,
        inference_config,
        prompt,
        negative_prompt,
        teacache_config,
        quantization_config,
        memory_config,
        lora_chain,
        talk_objects_config,
    ):
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
        return config

    def _save_image_if_needed(self, config, image):
        if config.task not in ["i2v", "s2v", "rs2v"]:
            return
        if image is None:
            raise ValueError("i2v or s2v or rs2v task requires input image")

        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        temp_path = self.temp_manager.create_temp_file(suffix=".png")
        pil_image.save(temp_path)
        config.image_path = temp_path
        logging.info(f"Image saved to {temp_path}")

    def _save_single_audio_if_needed(self, config, audio):
        # Route ComfyUI AUDIO straight into the runner via the in-memory shim
        # (no WAV temp file, no soundfile round-trip). The shim only intercepts
        # this single-AUDIO path; V3 multi-talker padding still produces real
        # files since its inputs are external paths/URLs, not ComfyUI tensors.
        if audio is None or not hasattr(config, "model_cls") or "seko" not in config.model_cls:
            return
        from ._audio_shim import comfyui_audio_to_loader_pair, install, register

        install()
        waveform, sr = comfyui_audio_to_loader_pair(audio)
        sentinel = register(waveform, sr)
        config.audio_path = sentinel
        logging.info(f"Routed ComfyUI AUDIO ({waveform.shape[0]}ch @ {sr}Hz, {waveform.shape[1]} samples) via in-memory shim")

    def _handle_talk_objects(self, config):
        if not getattr(config, "talk_objects", None):
            return

        src_objects, max_duration = self._resolve_talk_object_paths(config.talk_objects)
        processed_objects = self._process_talk_objects(src_objects, max_duration)
        self._commit_talk_objects(config, processed_objects)

    def _resolve_talk_object_paths(self, talk_objects):
        """Pull (audio, optional mask) per talker; resolve URLs and ComfyUI-relative paths.

        Always captures per-object duration so subclasses that pad can use it; V2 ignores it.
        Returns ``(src_objects, max_duration)``.
        """
        src_objects = []
        for talk_obj in talk_objects:
            obj = {}
            if "audio" in talk_obj:
                obj["audio"] = talk_obj["audio"]
            if "mask" in talk_obj:
                obj["mask"] = talk_obj["mask"]
            if "audio" in obj:
                src_objects.append(obj)

        max_duration = None
        for obj in src_objects:
            audio_path = obj.get("audio")
            if audio_path:
                obj["audio"] = self._resolve_one_asset(audio_path, kind="audio")
                if obj["audio"] and os.path.exists(obj["audio"]):
                    try:
                        duration = self._probe_audio_duration(obj["audio"])
                        obj["duration"] = duration
                        if max_duration is None or duration > max_duration:
                            max_duration = duration
                    except Exception as e:
                        logging.warning(f"Failed to probe audio duration for {obj['audio']}: {e}")

            mask_path = obj.get("mask")
            if mask_path:
                obj["mask"] = self._resolve_one_asset(mask_path, kind="mask")

        return src_objects, max_duration

    def _resolve_one_asset(self, path, kind):
        """Resolve URL → downloaded path; resolve ComfyUI-relative → absolute. Warn on missing."""
        if self.http_downloader.is_url(path):
            try:
                downloaded = self.http_downloader.download_if_url(path, prefix=kind)
                logging.info(f"Downloaded {kind} from URL: {path} -> {downloaded}")
                path = downloaded
            except Exception as e:
                logging.error(f"Failed to download {kind} from {path}: {e}")
                return path
        elif not os.path.isabs(path) and not path.startswith("/tmp"):
            resolved = self.resolver.resolve_input_path(path)
            logging.info(f"Resolved {kind} path: {path} -> {resolved}")
            path = resolved

        if not os.path.exists(path):
            logging.warning(f"{kind.capitalize()} file not found: {path}")
        return path

    @staticmethod
    def _probe_audio_duration(input_path: str) -> float:
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
        output = sp.check_output(cmd_probe, encoding="utf-8", errors="replace")
        data = json.loads(output)
        streams = data.get("streams", [])
        if not streams:
            raise ValueError(f"Failed to get audio stream information: {input_path}")
        return float(streams[0].get("duration", 0))

    def _commit_talk_objects(self, config, processed_objects):
        """Single talker w/o mask → set audio_path directly. Otherwise dump talk_objects.json."""
        if not processed_objects:
            return
        if len(processed_objects) == 1 and not processed_objects[0].get("mask", "").strip():
            config.audio_path = processed_objects[0]["audio"]
            logging.info(f"Convert Processed 1 talk object to audio path: {config.audio_path}")
            return

        temp_dir = self.temp_manager.create_temp_dir()
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump({"talk_objects": processed_objects}, f)
        config.audio_path = temp_dir
        logging.info(f"Processed {len(processed_objects)} talk objects")

    # --- hook for subclasses ---------------------------------------------

    def _process_talk_objects(self, src_objects, max_duration):
        """Default: pass through. V3 overrides this to pad + synthesize a bg talker."""
        return src_objects


class LightX2VConfigCombinerV2(_BaseConfigCombiner):
    """Aggregates configs and prepares image/audio/talk_objects. No multi-talker padding."""

    # Inherits everything; explicit no-op override here so the class isn't empty
    # and so the per-class identity / categorization stay distinct from V3.
    pass


class LightX2VConfigCombinerV3(_BaseConfigCombiner):
    """V2 + equal-duration audio padding and background-mask synthesis for multi-talker setups."""

    def _process_talk_objects(self, src_objects, max_duration):
        if len(src_objects) <= 1:
            return src_objects
        return self._pad_and_synthesize_bg(src_objects, max_duration)

    # --- V3-only multi-talker alignment ----------------------------------

    def _pad_and_synthesize_bg(self, src_objects, max_duration):
        """Pad each talker's audio to ``max_duration`` and append a (bg_audio, bg_mask) talker.

        The background talker carries silence-like white noise + a mask covering pixels
        that none of the per-speaker masks claim, so the runner has someone to "speak"
        for the rest of the frame.
        """
        processed = []
        mask_img_paths = []
        extend_count = 0

        for obj in src_objects:
            dst_obj = {"audio": obj["audio"]}
            src_audio_duration = obj.get("duration", max_duration)
            if max_duration - src_audio_duration > 0.1:
                dst_audio_path = self.temp_manager.create_temp_file(suffix=".mp3")
                self.extend_mp3(obj["audio"], dst_audio_path, max_duration)
                dst_obj["audio"] = dst_audio_path
                extend_count += 1
            src_mask = obj.get("mask")
            if src_mask:
                dst_obj["mask"] = src_mask
                mask_img_paths.append(src_mask)
            processed.append(dst_obj)
        logging.info(f"Extended {extend_count} audio files")

        bg_mask_io = self.generate_background_mask(mask_img_paths)
        bg_mask_path = self.temp_manager.create_temp_file(suffix=".jpg")
        with open(bg_mask_path, "wb") as f:
            f.write(bg_mask_io.getvalue())

        bg_noise = self.generate_white_noise(
            duration=max_duration,
            framerate=16000,
            n_channels=1,
            rms=0.00232,
            std_dev=0.00232,
        )
        wav_io = io.BytesIO()
        self.save_wav_file(audio_data=bg_noise, output_path=wav_io, framerate=16000, sample_width=2)
        bg_audio_path = self.temp_manager.create_temp_file(suffix=".wav")
        with open(bg_audio_path, "wb") as f:
            f.write(wav_io.getvalue())

        processed.append({"audio": bg_audio_path, "mask": bg_mask_path})
        logging.info(f"Generated background mask and audio: {bg_mask_path}, {bg_audio_path}")
        return processed

    # --- V3-only static utilities (kept here, not on the base) -----------

    @staticmethod
    def extend_mp3(input_path: str, output_path: str, duration: float) -> bool:
        """Pad audio to ``duration`` seconds; truncate if input is at most 0.1s longer.

        Errors if input exceeds duration by more than 0.1s.
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

    @staticmethod
    def generate_white_noise(
        duration: float,
        framerate: int,
        n_channels: int = 1,
        rms: float = None,
        std_dev: float = None,
        seed: int = None,
    ) -> np.ndarray:
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
        """White where all positive masks are ~zero (background), black elsewhere."""
        width = height = None
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
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            img_arrays.append(arr)

        threshold = 1
        zero_masks = []
        for arr in img_arrays:
            if arr.shape[-1] == 1:
                zero_mask = arr[:, :, 0] <= threshold
            else:
                zero_mask = np.all(arr <= threshold, axis=-1)
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
