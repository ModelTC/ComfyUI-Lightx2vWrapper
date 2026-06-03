"""Modular inference runner that consumes a PREPARED_CONFIG."""

import gc
import logging

import torch
from comfy.utils import ProgressBar

from ..config_builder import ConfigBuilder
from ..lightx2v.lightx2v.infer import init_runner
from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from ..lightx2v.lightx2v.utils.set_config import set_config


class LightX2VModularInferenceV2:
    """Pure inference node that takes prepared config and runs inference."""

    _current_runner = None
    _current_config_hash = None

    def __init__(self):
        if not hasattr(self.__class__, "_current_runner"):
            self.__class__._current_runner = None
        if not hasattr(self.__class__, "_current_config_hash"):
            self.__class__._current_config_hash = None

        self.config_builder = ConfigBuilder()

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

    def _get_config_hash(self, config) -> str:
        """Get hash of configuration to detect changes."""
        return self.config_builder.get_config_hash(config)

    def _build_rs2v_shot_config(self, config):
        from ..lightx2v.lightx2v.shot_runner.shot_base import load_clip_configs
        from ..lightx2v.lightx2v.utils.lockable_dict import LockableDict

        config_json = config.get("config_json")
        if config_json:
            main_cfg = config_json
        elif config.get("clip_configs"):
            main_cfg = config
        else:
            main_cfg = {
                "lightx2v_path": "",
                "clip_configs": [
                    {
                        "name": "rs2v_clip",
                        "config": LockableDict(config),
                    }
                ],
            }
            if "task" not in main_cfg["clip_configs"][0]["config"]:
                main_cfg["clip_configs"][0]["config"]["task"] = "rs2v"

        if isinstance(main_cfg, dict) and "lightx2v_path" not in main_cfg:
            main_cfg = dict(main_cfg)
            main_cfg["lightx2v_path"] = ""

        return load_clip_configs(main_cfg)

    def generate(self, prepared_config):
        """Run inference with prepared configuration."""

        config = prepared_config

        try:
            config_hash = self._get_config_hash(config)

            current_runner = getattr(self.__class__, "_current_runner", None)
            current_config_hash = getattr(self.__class__, "_current_config_hash", None)

            needs_reinit = current_runner is None or current_config_hash != config_hash or getattr(config, "lazy_load", False)

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if current_runner is not None:
                    del self.__class__._current_runner
                    torch.cuda.empty_cache()
                    gc.collect()
                if config.get("task") == "rs2v":
                    from ..lightx2v.lightx2v.shot_runner.rs2v_infer import ShotRS2VPipeline

                    shot_cfg = self._build_rs2v_shot_config(config)
                    self.__class__._current_runner = ShotRS2VPipeline(shot_cfg)
                else:
                    formatted_config = set_config(config)
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
                if hasattr(self.__class__, "_current_runner"):
                    del self.__class__._current_runner
                self.__class__._current_runner = None
                self.__class__._current_config_hash = None

            torch.cuda.empty_cache()
            gc.collect()

            return (images, audio)

        except Exception as e:
            logging.error(f"Error during inference: {e}")
            raise
