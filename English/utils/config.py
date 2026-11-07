import yaml


def load_config(path):
    """Read the single config.yaml and return (preprocess, model, train) dicts."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["preprocess"], cfg["model"], cfg["train"]
