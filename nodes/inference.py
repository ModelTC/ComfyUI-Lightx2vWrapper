"""Modular inference runner that consumes a PREPARED_CONFIG."""

import gc
import logging

import torch
from comfy.utils import ProgressBar

from ..config_builder import ConfigBuilder
from ..lightx2v.lightx2v.infer import init_runner
from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from ..lightx2v.lightx2v.utils.set_config import auto_calc_config, set_args2config


class LightX2VModularInferenceV2:
    """Pure inference node that takes prepared config and runs inference."""

    _current_runner = None
    _current_config_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_config": (
                    "PREPARED_CONFIG",
                    {"tooltip": "Fully prepared configuration from ConfigCombinerV2"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "generate"
    CATEGORY = "LightX2V/InferenceV2"

    @classmethod
    def _release_runner(cls):
        """Drop the singleton runner + force VRAM teardown via DefaultRunner.__del__.

        Callers MUST drop their own local refs to the old runner *before* invoking
        this — otherwise the refcount stays > 0, __del__ doesn't fire, and the
        next model load OOMs (model_a + model_b alive on GPU at the same time).
        """
        cls._current_runner = None
        cls._current_config_hash = None
        gc.collect()
        torch.cuda.empty_cache()

    def _build_rs2v_shot_config(self, config):
        from ..lightx2v.lightx2v.shot_runner.shot_base import load_clip_configs

        config_json = config.get("config_json")
        if config_json:
            main_cfg = config_json
        elif config.get("clip_configs"):
            main_cfg = config
        else:
            # load_clip_configs only runs set_config() on the "path" branch; for
            # in-memory clip configs we have to do it ourselves, otherwise framework
            # defaults (vae_stride, patch_size, ...) and the model's config.json
            # never get merged — rs2v_infer then KeyErrors on config["vae_stride"].
            if "task" not in config:
                config["task"] = "rs2v"
            # set_config = set_args2config + auto_calc_config. set_args2config strips
            # any key that's part of an InputInfo dataclass (target_video_length,
            # infer_steps, seed, ...). For CLI runs auto_calc_config recovers them
            # by merging --config_json, but we have no external JSON, so
            # auto_calc_config's `config["target_video_length"]` access KeyErrors.
            # Inject the bridge between set_args2config and auto_calc_config.
            target_video_length = config.get("target_video_length", config.get("segment_length", config.get("video_length", 81)))
            formatted = set_args2config(config)
            formatted["target_video_length"] = target_video_length
            formatted = auto_calc_config(formatted)
            main_cfg = {
                "lightx2v_path": "",
                "clip_configs": [
                    {
                        "name": "rs2v_clip",
                        "config": formatted,
                    }
                ],
            }

        if isinstance(main_cfg, dict) and "lightx2v_path" not in main_cfg:
            main_cfg = dict(main_cfg)
            main_cfg["lightx2v_path"] = ""

        return load_clip_configs(main_cfg)

    def generate(self, prepared_config):
        """Run inference with prepared configuration."""

        config = prepared_config
        # Combiner may have stashed the input AUDIO in the in-memory shim and
        # set audio_path to a sentinel — release it on exit so the tensor isn't
        # retained across runs (one ComfyUI graph tick = one sentinel).
        from ._audio_shim import is_sentinel as _is_audio_sentinel
        from ._audio_shim import release as _release_audio_sentinel

        _audio_sentinel = config.get("audio_path") if isinstance(config, dict) else getattr(config, "audio_path", None)
        if not _is_audio_sentinel(_audio_sentinel):
            _audio_sentinel = None

        try:
            config_hash = ConfigBuilder.get_config_hash(config)

            current_runner = getattr(self.__class__, "_current_runner", None)
            current_config_hash = getattr(self.__class__, "_current_config_hash", None)

            needs_reinit = current_runner is None or current_config_hash != config_hash or getattr(config, "lazy_load", False)

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if current_runner is not None:
                    # Free old runner VRAM BEFORE constructing the new one, otherwise
                    # both models live on GPU during the second load -> OOM (seen when
                    # switching v2.5 s2v -> v2.7 rs2v).
                    current_runner = None
                    self._release_runner()
                if config.get("task") == "rs2v":
                    from ..lightx2v.lightx2v.shot_runner.rs2v_infer import ShotRS2VPipeline

                    shot_cfg = self._build_rs2v_shot_config(config)
                    self.__class__._current_runner = ShotRS2VPipeline(shot_cfg)
                else:
                    # set_args2config strips InputInfo-dataclass keys (target_video_length,
                    # infer_steps, ...). CLI flows recover them via --config_json merging
                    # inside auto_calc_config; our in-memory flow has no external JSON, so
                    # we bridge target_video_length manually between the two halves so
                    # auto_calc_config:194's modulo check on s2v/i2v doesn't KeyError.
                    target_video_length = config.get(
                        "target_video_length",
                        config.get("segment_length", config.get("video_length", 81)),
                    )
                    formatted_config = set_args2config(config)
                    formatted_config["target_video_length"] = target_video_length
                    formatted_config = auto_calc_config(formatted_config)
                    self.__class__._current_runner = init_runner(formatted_config)
                self.__class__._current_config_hash = config_hash

            progress = ProgressBar(100)

            def update_progress(current_step, _total):
                progress.update_absolute(current_step)

            current_runner = self.__class__._current_runner

            if hasattr(current_runner, "set_progress_callback"):
                current_runner.set_progress_callback(update_progress)

            config["return_result_tensor"] = True
            config["save_result_path"] = ""
            config["negative_prompt"] = config.get("negative_prompt", "")
            if config.get("task") == "rs2v":
                result_dict = current_runner.run_pipeline(config)
            else:
                input_data = init_empty_input_info(config.task)
                update_input_info_from_dict(input_data, config)
                current_runner.set_config(config)
                result_dict = current_runner.run_pipeline(input_data)

            images = result_dict.get("video", None)
            audio = result_dict.get("audio", None)

            if images is not None and images.numel() > 0:
                images = images.cpu()
                if images.dtype != torch.float32:
                    images = images.float()

            if getattr(config, "unload_after_inference", False):
                current_runner = None  # drop local ref so __del__ can run
                self._release_runner()
            else:
                torch.cuda.empty_cache()
                gc.collect()

            return (images, audio)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise
        finally:
            if _audio_sentinel is not None:
                _release_audio_sentinel(_audio_sentinel)
