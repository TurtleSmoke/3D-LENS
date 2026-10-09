import logging
from pathlib import Path

from omegaconf import OmegaConf

from reid.configs import default_config

logger = logging.getLogger("reid.configs")


def get_cfg(args):
    """Merge order is default_config.yml -> --config file -> --opts."""
    opts = args.opts + [
        f"GLOBAL.NAME={args.name}",
        f"GLOBAL.OUTPUT_DIR={args.output_dir}",
        f"DATA.ROOT_DIR={args.root_dir}",
        f"GLOBAL.MODE={args.mode}",
    ]
    return OmegaConf.merge(default_config, OmegaConf.load(args.config), OmegaConf.from_cli(opts))


def setup(args):
    cfg = get_cfg(args)
    cfg_dir = Path(cfg.GLOBAL.OUTPUT_DIR) / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dumping config to {cfg_dir / (cfg.GLOBAL.NAME + '.yaml')}")
    OmegaConf.save(config=cfg, f=cfg_dir / f"{cfg.GLOBAL.NAME}.yaml")
    return cfg
