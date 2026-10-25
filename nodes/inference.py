"""Modular inference runner that consumes a PREPARED_CONFIG."""

import gc
import logging

import torch
from comfy.utils import ProgressBar

from ..config_builder import ConfigBuilder

# LightX2V 0.6 replaced the resident ``init_runner`` entrypoint and the
# ``set_args2config``/``auto_calc_config`` pair with a startup/request split.
# Keep the imports optional so the wrapper can select the API at runtime while
# still supporting the previous submodule revision.
try:
    from ..lightx2v.lightx2v.infer import init_runner as _legacy_init_runner
except ImportError:
    _legacy_init_runner = None

try:
    from ..lightx2v.lightx2v.utils.input_info import init_empty_input_info as _legacy_init_empty_input_info
    from ..lightx2v.lightx2v.utils.input_info import update_input_info_from_dict as _legacy_update_input_info_from_dict
except ImportError:
    _legacy_init_empty_input_info = None
    _legacy_update_input_info_from_dict = None

try:
    from ..lightx2v.lightx2v.utils.set_config import auto_calc_config as _legacy_auto_calc_config
    from ..lightx2v.lightx2v.utils.set_config import set_args2config as _legacy_set_args2config
except ImportError:
    _legacy_auto_calc_config = None
    _legacy_set_args2config = None

try:
    from ..lightx2v.lightx2v.models.runners.runner_factory import build_runner as _modern_build_runner
    from ..lightx2v.lightx2v.utils.set_config import build_startup_config as _modern_build_startup_config
except ImportError:
    _modern_build_runner = None
    _modern_build_startup_config = None


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
        """Drop the singleton runner and its GPU modules before another load.

        Callers MUST drop their own local refs to the old runner *before* invoking
        this — otherwise the refcount stays > 0, __del__ doesn't fire, and the
        next model load OOMs (model_a + model_b alive on GPU at the same time).
        """
        from ._runner_memory import dispose_runner

        gc.unfreeze()
        dispose_runner(cls._current_runner)
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
            # LightX2V 0.6 uses the same startup builder as the regular runner;
            # older revisions need the set_args2config + auto_calc_config pair.
            # The shot runner's load_clip_configs() calls build_startup_config
            # again, so passing the modern startup shape here is safe and keeps
            # model config/vae metadata loading in LightX2V itself.
            if _modern_build_startup_config is not None:
                startup_data = dict(config)
                startup_data["size"] = [
                    int(config.get("target_height", config.get("height", 480))),
                    int(config.get("target_width", config.get("width", 832))),
                ]
                startup_data["num_frames"] = int(config.get("target_video_length", config.get("video_length", config.get("num_frames", 81))))
                formatted = _modern_build_startup_config(startup_data)
            else:
                # set_args2config strips any key that's part of an InputInfo
                # dataclass (target_video_length, infer_steps, seed, ...). For
                # CLI runs auto_calc_config recovers them by merging
                # --config_json; our in-memory flow has no external JSON, so
                # bridge target_video_length back before auto_calc_config.
                target_video_length = config.get("target_video_length", config.get("segment_length", config.get("video_length", 81)))
                if _legacy_set_args2config is None or _legacy_auto_calc_config is None:
                    raise RuntimeError("No compatible LightX2V shot-runner config API found")
                formatted = _legacy_set_args2config(config)
                formatted["target_video_length"] = target_video_length
                formatted = _legacy_auto_calc_config(formatted)
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

    @staticmethod
    def _build_modern_startup_config(config):
        """Translate wrapper shape fields into LightX2V 0.6 startup fields."""
        if _modern_build_startup_config is None or _modern_build_runner is None:
            raise RuntimeError("The installed LightX2V revision does not provide the modern startup API")

        startup_data = dict(config)
        startup_data["size"] = [
            int(config.get("target_height", config.get("height", 480))),
            int(config.get("target_width", config.get("width", 832))),
        ]
        startup_data["num_frames"] = int(config.get("target_video_length", config.get("video_length", config.get("num_frames", 81))))
        return _modern_build_startup_config(startup_data)

    @staticmethod
    def _build_modern_request(config, runner):
        """Build only fields accepted by the current runner's request schema."""
        task = config.get("task")
        supported_fields = runner.get_supported_request_fields(task)
        request = {}
        for field in supported_fields:
            if field == "size":
                # SekoTalk's fixed-area/min-side modes derive the output size
                # from the model's resize policy.  The 0.6 request validator
                # rejects an explicit size unless the request selected
                # fixed_shape; keep the wrapper's target dimensions in startup
                # config but do not send them as a per-request override.
                if config.get("model_cls", "").startswith("seko") and config.get("resize_mode") != "fixed_shape":
                    continue
                request[field] = [
                    int(config.get("target_height", config.get("height", 480))),
                    int(config.get("target_width", config.get("width", 832))),
                ]
            elif field == "num_frames":
                request[field] = int(config.get("target_video_length", config.get("video_length", config.get("num_frames", 81))))
            elif field in config:
                request[field] = config[field]

        request["task"] = task
        request["return_result_tensor"] = True
        request["save_result_path"] = ""
        return request

    def generate(self, prepared_config):
        """Run inference with prepared configuration."""

        config = prepared_config
        from ._image_shim import install_on_runner_tree as _install_image_shim
        from ._image_shim import is_sentinel as _is_image_sentinel
        from ._image_shim import release as _release_image_sentinel

        _image_sentinel = config.get("image_path") if isinstance(config, dict) else getattr(config, "image_path", None)
        if not _is_image_sentinel(_image_sentinel):
            _image_sentinel = None

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

            needs_reinit = current_runner is None or current_config_hash != config_hash or config.get("lazy_load", False)

            logging.info(f"Needs reinit: {needs_reinit}, old config hash: {current_config_hash}, new config hash: {config_hash}")
            if needs_reinit:
                if current_runner is not None:
                    # Free old runner VRAM BEFORE constructing the new one, otherwise
                    # both models live on GPU during the second load -> OOM (seen when
                    # switching v2.5 s2v -> v2.7 rs2v).
                    current_runner = None
                    self._release_runner()
                # ComfyUI may still cache SwiftVR/SeedVR loader outputs from a
                # prior workflow.  Evict those GPU runners before loading a
                # Wan/SekoTalk model, while retaining rebuildable handles.
                from ._runner_memory import release_restoration_runners, unload_comfyui_models

                release_restoration_runners()
                unload_comfyui_models()
                if config.get("task") == "rs2v":
                    from ..lightx2v.lightx2v.shot_runner.rs2v_infer import ShotRS2VPipeline

                    shot_cfg = self._build_rs2v_shot_config(config)
                    self.__class__._current_runner = ShotRS2VPipeline(shot_cfg)
                else:
                    if _modern_build_runner is not None and _modern_build_startup_config is not None:
                        startup_config = self._build_modern_startup_config(config)
                        self.__class__._current_runner = _modern_build_runner(startup_config)
                    else:
                        if _legacy_set_args2config is None or _legacy_auto_calc_config is None or _legacy_init_runner is None:
                            raise RuntimeError("No compatible LightX2V runner API found in the installed LightX2V submodule")
                        # The legacy API strips InputInfo-dataclass keys.  Bridge
                        # target_video_length back before auto_calc_config.
                        target_video_length = config.get(
                            "target_video_length",
                            config.get("segment_length", config.get("video_length", 81)),
                        )
                        formatted_config = _legacy_set_args2config(config)
                        formatted_config["target_video_length"] = target_video_length
                        formatted_config = _legacy_auto_calc_config(formatted_config)
                        self.__class__._current_runner = _legacy_init_runner(formatted_config)
                self.__class__._current_config_hash = config_hash

            progress = ProgressBar(100)

            def update_progress(current_step, _total):
                progress.update_absolute(current_step)

            current_runner = self.__class__._current_runner

            if _image_sentinel is not None:
                if _install_image_shim(current_runner) == 0:
                    raise RuntimeError("LightX2V IMAGE input shim could not find a runner image reader")

            if hasattr(current_runner, "set_progress_callback"):
                current_runner.set_progress_callback(update_progress)

            config["return_result_tensor"] = True
            config["save_result_path"] = ""
            config["negative_prompt"] = config.get("negative_prompt", "")
            from ._runner_memory import active_lightx2v_request

            with active_lightx2v_request():
                if config.get("task") == "rs2v":
                    # ShotRS2VPipeline delegates request validation to its clip
                    # runner. Filter prepared config against the request schema.
                    if _modern_build_runner is not None and _modern_build_startup_config is not None:
                        shot_runner = current_runner.clip_generators["rs2v_clip"]
                        request_data = self._build_modern_request(config, shot_runner)
                        result_dict = current_runner.run_pipeline(request_data)
                    else:
                        result_dict = current_runner.run_pipeline(config)
                elif _modern_build_runner is not None and _modern_build_startup_config is not None:
                    request_data = self._build_modern_request(config, current_runner)
                    input_info = current_runner.prepare_request(request_data)
                    result_dict = current_runner.run_request(input_info)
                else:
                    if _legacy_init_empty_input_info is None or _legacy_update_input_info_from_dict is None:
                        raise RuntimeError("No compatible LightX2V input-info API found in the installed LightX2V submodule")
                    input_data = _legacy_init_empty_input_info(config.task)
                    _legacy_update_input_info_from_dict(input_data, config)
                    current_runner.set_config(config)
                    result_dict = current_runner.run_pipeline(input_data)

            images = result_dict.get("video", None)
            audio = result_dict.get("audio", None)

            if images is not None and images.numel() > 0:
                images = images.cpu()
                if images.dtype != torch.float32:
                    images = images.float()

            if config.get("unload_after_inference", False):
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
            if _image_sentinel is not None:
                _release_image_sentinel(_image_sentinel)
            if _audio_sentinel is not None:
                _release_audio_sentinel(_audio_sentinel)
