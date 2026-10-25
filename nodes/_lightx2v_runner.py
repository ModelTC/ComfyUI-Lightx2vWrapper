"""Small adapters for LightX2V 0.6's startup/request runner API."""

from contextlib import contextmanager


def build_runner(startup_data):
    """Build a resident LightX2V runner from deployment-time settings."""
    from ..lightx2v.lightx2v.models.runners.runner_factory import build_runner as _build_runner
    from ..lightx2v.lightx2v.utils.set_config import build_startup_config

    startup_config = build_startup_config(dict(startup_data))
    return _build_runner(startup_config)


@contextmanager
def temporary_runner_config(runner, values):
    """Apply per-call implementation settings while a request is running.

    LightX2V 0.6 keeps deployment settings in a locked ``LockableDict`` and
    exposes request data through ``prepare_request``.  A few legacy ComfyUI
    controls (SeedVR segmenting/color correction and SwiftVR video encoding)
    are still implementation settings, so they need a narrow, serialized
    override instead of being passed as unsupported request fields.
    """
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        yield
        return

    config = runner.config
    missing = object()
    previous = {key: config.get(key, missing) for key in values}
    unlock = getattr(config, "temporarily_unlocked", None)
    if unlock is None:
        raise RuntimeError("LightX2V runner config does not support temporary request settings")

    with unlock():
        config.update(values)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is missing:
                    config.pop(key, None)
                else:
                    config[key] = value


def run_request(runner, request, *, runtime_config=None):
    """Validate one modern request and execute it through the runner API."""
    from ._runner_memory import active_lightx2v_request

    with active_lightx2v_request(), temporary_runner_config(runner, runtime_config or {}):
        input_info = runner.prepare_request(dict(request))
        return runner.run_request(input_info)
