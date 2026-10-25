"""Regression checks for switching cached LightX2V model outputs."""

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


class RunnerMemoryTests(unittest.TestCase):
    def setUp(self):
        package = "_runner_memory_test_package"
        nodes_package = package + ".nodes"
        self.build_runner = Mock(side_effect=lambda config: {"model_cls": config["model_cls"]})
        self.release_modular = Mock()
        self.empty_cache = Mock()
        self.unload_native = Mock()
        self.soft_empty_cache = Mock()
        self.native_free_memory = Mock(return_value=["native-unloaded"])
        self.modular_node = types.SimpleNamespace(_current_runner=None, _release_runner=self.release_modular)

        modules = {
            package: types.ModuleType(package),
            nodes_package: types.ModuleType(nodes_package),
            nodes_package + "._lightx2v_runner": types.ModuleType(nodes_package + "._lightx2v_runner"),
            nodes_package + ".inference": types.ModuleType(nodes_package + ".inference"),
            "torch": types.ModuleType("torch"),
            "comfy": types.ModuleType("comfy"),
            "comfy.model_management": types.ModuleType("comfy.model_management"),
        }
        modules[nodes_package + "._lightx2v_runner"].build_runner = self.build_runner
        modules[nodes_package + ".inference"].LightX2VModularInferenceV2 = self.modular_node
        modules["torch"].cuda = types.SimpleNamespace(empty_cache=self.empty_cache)
        modules["comfy"].model_management = modules["comfy.model_management"]
        modules["comfy.model_management"].unload_all_models = self.unload_native
        modules["comfy.model_management"].soft_empty_cache = self.soft_empty_cache
        modules["comfy.model_management"].free_memory = self.native_free_memory
        self.saved_modules = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)

        path = Path(__file__).resolve().parents[1] / "nodes" / "_runner_memory.py"
        name = nodes_package + "._runner_memory"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.module = module
        self.module_name = name

    def tearDown(self):
        sys.modules.pop(self.module_name, None)
        for name, previous in self.saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_switch_evicts_runner_from_cached_loader_output(self):
        first = self.module.make_runner_handle({"model_cls": "seedvr2"})
        second = self.module.make_runner_handle({"model_cls": "swiftvr"})

        self.assertIsNone(first["runner"])
        self.assertEqual(second["runner"]["model_cls"], "swiftvr")
        self.assertEqual(self.unload_native.call_count, 2)
        self.assertGreaterEqual(self.empty_cache.call_count, 2)

        restored = self.module.get_runner(first)
        self.assertEqual(restored["model_cls"], "seedvr2")
        self.assertIsNone(second["runner"])
        self.assertEqual(self.build_runner.call_count, 3)
        self.assertEqual(self.unload_native.call_count, 3)

    def test_native_memory_request_evicts_lightx2v_runner(self):
        handle = self.module.make_runner_handle({"model_cls": "swiftvr"})
        self.module.install_comfyui_memory_hook()
        self.module.install_comfyui_memory_hook()
        result = sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cuda"))

        self.assertEqual(result, ["native-unloaded"])
        self.assertIsNone(handle["runner"])
        self.native_free_memory.assert_called_once()

    def test_active_lightx2v_request_cannot_be_evicted(self):
        handle = self.module.make_runner_handle({"model_cls": "seedvr2"})
        self.module.install_comfyui_memory_hook()
        with self.module.active_lightx2v_request():
            sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cuda"))
            self.assertIsNotNone(handle["runner"])
        sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cuda"))
        self.assertIsNone(handle["runner"])

    def test_native_memory_request_evicts_modular_runner(self):
        self.modular_node._current_runner = types.SimpleNamespace(model=object())
        self.module.install_comfyui_memory_hook()
        sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cuda"))
        self.release_modular.assert_called_once_with()

    def test_native_memory_hook_does_not_reenter_during_teardown(self):
        self.modular_node._current_runner = types.SimpleNamespace(model=object())
        self.module.install_comfyui_memory_hook()
        model_management = sys.modules["comfy.model_management"]
        self.release_modular.side_effect = lambda: model_management.free_memory(100, types.SimpleNamespace(type="cuda"))
        model_management.free_memory(100, types.SimpleNamespace(type="cuda"))
        self.release_modular.assert_called_once_with()

    def test_concurrent_native_load_waits_for_lightx2v_request(self):
        handle = self.module.make_runner_handle({"model_cls": "swiftvr"})
        self.module.install_comfyui_memory_hook()
        started = threading.Event()
        finished = threading.Event()

        def load_native():
            started.set()
            sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cuda"))
            finished.set()

        with self.module.active_lightx2v_request():
            thread = threading.Thread(target=load_native)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(finished.wait(0.05))
            self.assertIsNotNone(handle["runner"])
        thread.join(timeout=1)
        self.assertTrue(finished.is_set())
        self.assertIsNone(handle["runner"])

    def test_cpu_memory_request_keeps_lightx2v_runner(self):
        handle = self.module.make_runner_handle({"model_cls": "swiftvr"})
        self.module.install_comfyui_memory_hook()
        sys.modules["comfy.model_management"].free_memory(100, types.SimpleNamespace(type="cpu"))
        self.assertIsNotNone(handle["runner"])

    def test_modular_switch_releases_restoration_runner(self):
        cached_output = self.module.make_runner_handle({"model_cls": "swiftvr"})
        self.module.release_restoration_runners()

        self.assertIsNone(cached_output["runner"])
        self.assertEqual(cached_output.startup_config, {"model_cls": "swiftvr"})

    def test_seedvr_switch_clears_bound_method_cache(self):
        clear = Mock()
        name = "lightx2v.models.networks.seedvr.utils.rope"
        previous = sys.modules.get(name)
        rope_module = types.ModuleType(name)
        rope_module.RotaryEmbeddingBase = types.SimpleNamespace(get_axial_freqs=types.SimpleNamespace(cache_clear=clear))
        sys.modules[name] = rope_module
        try:
            cached_output = self.module.make_runner_handle({"model_cls": "seedvr2"})
            self.module.make_runner_handle({"model_cls": "swiftvr"})
            self.assertIsNone(cached_output["runner"])
            clear.assert_called_once_with()
        finally:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_dispose_runner_drops_swiftvr_restorer(self):
        runner = types.SimpleNamespace(model=object(), restorer=object(), vae_decoder=object())
        self.module.dispose_runner(runner)
        self.assertFalse(hasattr(runner, "model"))
        self.assertFalse(hasattr(runner, "restorer"))
        self.assertFalse(hasattr(runner, "vae_decoder"))

    def test_dispose_runner_recurses_into_rs2v_clips(self):
        child = types.SimpleNamespace(model=object(), vae_decoder=object())
        shot = types.SimpleNamespace(clip_generators={"rs2v_clip": child})
        self.module.dispose_runner(shot)
        self.assertFalse(hasattr(shot, "clip_generators"))
        self.assertFalse(hasattr(child, "model"))


if __name__ == "__main__":
    unittest.main()
