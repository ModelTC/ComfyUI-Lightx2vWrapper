# Contributing

## Development setup

The wrapper follows the Python version supported by the bundled LightX2V
submodule (Python 3.10 or newer).

```bash
git clone --recursive https://github.com/ModelTC/ComfyUI-Lightx2vWrapper.git
cd ComfyUI-Lightx2vWrapper
python -m pip install -r lightx2v/requirements.txt
python -m pip install -r requirements.txt pre-commit
pre-commit install
```

## Before opening a pull request

Run the repository checks from the project root:

```bash
pre-commit run --all-files
```

For workflow or runner changes, validate the affected ComfyUI workflow with a
local LightX2V installation and include the workflow or reproduction details
in the pull request description. Keep changes to the `lightx2v` submodule in a
separate upstream contribution unless the wrapper specifically requires a
submodule revision update.

Pull requests should explain the user-visible behavior, list validation
performed, and avoid committing model weights, generated media, credentials,
or local debugging artifacts.
