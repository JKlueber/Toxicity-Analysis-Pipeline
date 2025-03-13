import yaml
from pathlib import Path

def load_config():
    with Path("src/config/config.yaml").open('r') as file:
        config = yaml.safe_load(file)
    return config
