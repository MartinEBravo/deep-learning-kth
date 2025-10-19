from pathlib import Path


def ensure_dir(path):
    """
    Ensure the provided directory exists before saving artefacts.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
