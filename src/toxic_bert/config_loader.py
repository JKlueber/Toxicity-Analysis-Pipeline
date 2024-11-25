import yaml
from pathlib import Path

def load_config(config_path: Path):
    with config_path.open('r') as file:
        config = yaml.safe_load(file)
    return config
