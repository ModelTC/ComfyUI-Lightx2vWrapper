"""Keep at most one resident LightX2V runner across ComfyUI node caches.

ComfyUI can retain loader outputs after a workflow stops using them.  Holding a
runner directly in each output therefore keeps old models on the GPU when the
next workflow loads another model.  Handles remain cacheable, but their runner
slot can be cleared and rebuilt when that workflow is used again.
"""

import functools
import gc
import logging
import sys
import threading
import weakref
from contextlib import contextmanager

import torch

from ._lightx2v_runner import build_runner

_LOCK = threading.RLock()
_HANDLES = weakref.WeakValueDictionary()
_ACTIVE_REQUESTS = 0
_RELEASING = False


@contextmanager
def active_lightx2v_request():
    """Keep ComfyUI's memory hook from evicting a runner during inference."""
    global _ACTIVE_REQUESTS
    with _LOCK:
        _ACTIVE_REQUESTS += 1
        try:
            yield
        finally:
            _ACTIVE_REQUESTS -= 1


def dispose_runner(runner):
    """Drop GPU modules even when an upstream callback still references runner."""
    if runner is None:
        return
    children = getattr(runner, "clip_generators", None)
    if isinstance(children, dict):
        for child in children.values():
            dispose_runner(child)
        del runner.clip_generators
    # DefaultRunner.__del__ covers most of these, but only runs once *all*
    # external references disappear.  SwiftVR's restorer also holds its own
    # transformer and VAE references and is not covered by that destructor.
    for name in ("restorer", "model", "text_encoders", "image_encoder", "vae_encoder", "vae_decoder", "inputs"):
        if name in getattr(runner, "__dict__", {}):
            delattr(runner, name)


def _collect_cuda():
    # LightX2V freezes the GC generation after a successful request.  Unfreeze
    # before dropping runners so cycles containing CUDA tensors can be collected.
    gc.unfreeze()
    gc.collect()
    torch.cuda.empty_cache()


def release_restoration_runners(keep=None):
    """Drop SwiftVR/SeedVR runners, including cached ComfyUI loader outputs."""
    with _LOCK:
        gc.unfreeze()
        released_seedvr = False
        released_swiftvr = False
        for handle in list(_HANDLES.values()):
            if handle is not keep:
                runner = handle["runner"]
                released_seedvr |= runner is not None and handle.startup_config.get("model_cls") == "seedvr2"
                released_swiftvr |= runner is not None and handle.startup_config.get("model_cls") == "swiftvr"
                dispose_runner(runner)
                handle["runner"] = None
                del runner
        if released_seedvr:
            # This @lru_cache is keyed by `self`: without clearing it, every
            # SeedVR RoPE instance (and its GPU model graph) remains reachable.
            rope_module = sys.modules.get("lightx2v.models.networks.seedvr.utils.rope")
            if rope_module is not None:
                rope_module.RotaryEmbeddingBase.get_axial_freqs.cache_clear()
        if released_swiftvr:
            attention_module = sys.modules.get("lightx2v.models.networks.swiftvr.attention")
            if attention_module is not None:
                attention_module.WindowLayout.clear()
        _collect_cuda()


def _release_modular_runner():
    from .inference import LightX2VModularInferenceV2

    if LightX2VModularInferenceV2._current_runner is not None:
        LightX2VModularInferenceV2._release_runner()


def release_all_runners():
    """Evict every resident LightX2V model before ComfyUI allocates VRAM."""
    global _RELEASING
    with _LOCK:
        if _ACTIVE_REQUESTS or _RELEASING:
            return False
        from .inference import LightX2VModularInferenceV2

        if LightX2VModularInferenceV2._current_runner is None and not any(handle["runner"] is not None for handle in _HANDLES.values()):
            return False
        _RELEASING = True
        try:
            logging.info("Releasing resident LightX2V models for ComfyUI memory management")
            _release_modular_runner()
            release_restoration_runners()
            return True
        finally:
            _RELEASING = False


def unload_comfyui_models():
    """Release ComfyUI-owned models before LightX2V constructs a GPU runner."""
    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def install_comfyui_memory_hook():
    """Evict LightX2V when ComfyUI's native model manager frees GPU memory.

    Native loads call free_memory through the model_management module, so this
    also covers call sites holding an early-bound reference to load_models_gpu.
    """
    import comfy.model_management as model_management

    original = model_management.free_memory
    if getattr(original, "_lightx2v_memory_hook", False) is True:
        return

    @functools.wraps(original)
    def free_memory_with_lightx2v(*args, **kwargs):
        device = args[1] if len(args) > 1 else kwargs.get("device")
        if device is None or getattr(device, "type", None) != "cpu":
            release_all_runners()
        return original(*args, **kwargs)

    free_memory_with_lightx2v._lightx2v_memory_hook = True
    model_management.free_memory = free_memory_with_lightx2v


def make_runner_handle(config, **metadata):
    """Build a runner after releasing every other LightX2V model."""
    with _LOCK:
        release_restoration_runners()
        _release_modular_runner()
        unload_comfyui_models()
        runner = build_runner(config)
        handle = RunnerHandle(config, runner, **metadata)
        _HANDLES[id(handle)] = handle
        return handle


def get_runner(model):
    """Rebuild an evicted cached loader output before a sampler uses it."""
    if not isinstance(model, RunnerHandle):
        return model["runner"]
    with _LOCK:
        runner = model["runner"]
        if runner is None:
            release_restoration_runners(keep=model)
            _release_modular_runner()
            unload_comfyui_models()
            runner = build_runner(model.startup_config)
            model["runner"] = runner
        return runner


class RunnerHandle(dict):
    """ComfyUI-compatible model output whose GPU state can be evicted."""

    def __init__(self, startup_config, runner, **metadata):
        super().__init__(runner=runner, **metadata)
        self.startup_config = dict(startup_config)
