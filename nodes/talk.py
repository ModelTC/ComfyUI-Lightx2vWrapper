"""Talk-object input and combiner nodes (for multi-speaker audio-driven generation)."""

from ..config_builder import TalkObjectConfigBuilder
from ..data_models import TalkObjectsConfig


class TalkObjectInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (
                    "STRING",
                    {"default": "person_1", "tooltip": "speaker name identifier"},
                ),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "uploaded audio file"}),
                "mask": ("MASK", {"tooltip": "uploaded mask image (optional)"}),
                "save_to_input": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "save to input folder"},
                ),
            },
        }

    RETURN_TYPES = ("TALK_OBJECT",)
    RETURN_NAMES = ("talk_object",)
    FUNCTION = "create_talk_object"
    CATEGORY = "LightX2V/Audio"

    def create_talk_object(self, name, audio=None, mask=None, save_to_input=True):
        """Create a talk object from input data."""
        builder = TalkObjectConfigBuilder()

        talk_object = builder.build_from_input(name=name, audio=audio, mask=mask, save_to_input=save_to_input)

        if talk_object:
            return (talk_object,)
        return (None,)


class TalkObjectsCombiner:
    PREDEFINED_SLOTS = 16

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}, "optional": {}}

        for i in range(cls.PREDEFINED_SLOTS):
            inputs["optional"][f"talk_object_{i + 1}"] = (
                "TALK_OBJECT",
                {"tooltip": f"talk object {i + 1}"},
            )

        return inputs

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "combine_talk_objects"
    CATEGORY = "LightX2V/Audio"

    def combine_talk_objects(self, **kwargs):
        config = TalkObjectsConfig()

        for i in range(self.PREDEFINED_SLOTS):
            talk_obj = kwargs.get(f"talk_object_{i + 1}")

            if talk_obj is not None:
                config.add_object(talk_obj)

        if not config.talk_objects:
            return (None,)

        return (config,)


class TalkObjectsFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_config": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '[{"name": "person1", "audio": "/path/to/audio1.wav", "mask": "/path/to/mask1.png"}]',
                        "tooltip": "JSON format talk objects configuration",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "parse_json_config"
    CATEGORY = "LightX2V/Audio"

    def parse_json_config(self, json_config):
        builder = TalkObjectConfigBuilder()
        talk_objects_config = builder.build_from_json(json_config)
        return (talk_objects_config,)


class TalkObjectsFromFiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "audio1.wav\naudio2.wav",
                        "tooltip": "audio file list (one per line)",
                    },
                ),
            },
            "optional": {
                "mask_files": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "mask1.png\nmask2.png",
                        "tooltip": "mask file list (one per line, optional)",
                    },
                ),
                "names": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "person1\nperson2",
                        "tooltip": "talk object name list (one per line, optional)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TALK_OBJECTS_CONFIG",)
    RETURN_NAMES = ("talk_objects_config",)
    FUNCTION = "build_from_files"
    CATEGORY = "LightX2V/Audio"

    def build_from_files(self, audio_files, mask_files="", names=""):
        builder = TalkObjectConfigBuilder()
        talk_objects_config = builder.build_from_files(audio_files, mask_files, names)
        return (talk_objects_config,)
