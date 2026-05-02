from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()