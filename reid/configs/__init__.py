import pathlib

from omegaconf import OmegaConf


def load_config(config_name: str):
    config_path = pathlib.Path(__file__).parent / f"{config_name}.yml"
    return OmegaConf.load(config_path)


default_config = load_config("default_config")
