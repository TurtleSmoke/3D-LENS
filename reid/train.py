import argparse
import logging
import shutil
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from reid.configs.config import setup
from reid.data import DataManager
from reid.evaluate import evaluate
from reid.loss.make_loss import loss_func
from reid.metrics.meter import AverageMeter
from reid.models.networks import ReID
from reid.optim import build_optimizer
from reid.optim.lr_scheduler import build_scheduler
from reid.utils.utils import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reid.train")


def get_args_parser():
    parser = argparse.ArgumentParser("3D-LENS training", add_help=True)
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--root_dir", type=str, required=True, help="Directory containing the datasets")
    parser.add_argument("--mode", type=str, choices=["debug", "release"], default="release", help="Run mode")
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help='Modify config options using space-separated "PATH.KEY=VALUE" pairs.',
    )
    return parser


def train(cfg):
    if not cfg.MODEL.REID.get("PRETRAINED_PATH"):
        raise ValueError("MODEL.REID.PRETRAINED_PATH (ImageNet-21k ViT-B/16) is required for training")
    data_manager = DataManager(cfg)
    data_manager.summary()

    model = ReID(cfg.MODEL.REID, data_manager.train_loader.dataset.num_classes).to("cuda")

    iters_per_epoch = len(data_manager.train_loader)
    optimizer = build_optimizer(model, cfg.TRAIN.OPTIMIZER)
    scheduler = build_scheduler(optimizer, cfg.TRAIN.SCHEDULER, iters_per_epoch)

    tensorboard_dir = Path(cfg.GLOBAL.OUTPUT_DIR) / "tensorboard_logs" / cfg.GLOBAL.NAME
    shutil.rmtree(tensorboard_dir, ignore_errors=True)
    writer = SummaryWriter(tensorboard_dir)
    save_dir = Path(cfg.GLOBAL.OUTPUT_DIR) / "ckpts" / cfg.GLOBAL.NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler("cuda")
    steps = 0
    data_loader_iter = iter(data_manager.train_loader)

    for epoch in range(1, cfg.TRAIN.MAX_EPOCHS + 1):
        logger.info(f"Epoch {epoch} / {cfg.TRAIN.MAX_EPOCHS}")
        # Curriculum, the rejection probability of hard synthetic views depends on the epoch
        if hasattr(data_manager.train_sampler, "set_epoch"):
            data_manager.train_sampler.set_epoch(epoch)

        start_time = time.time()
        loss_meter.reset()
        model.train()

        for n_iter in range(iters_per_epoch):
            if cfg.GLOBAL.MODE == "debug" and n_iter > 5:
                break
            data = next(data_loader_iter)
            scale = scaler.get_scale()

            x = data["x"].to("cuda")
            target = data["pid"].to("cuda")
            domain_labels = data["domain_label"].to("cuda")

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(x)
                losses = loss_func(outputs, target, domain_labels, cfg.TRAIN)

            optimizer.zero_grad()
            scaler.scale(losses["TOTAL_LOSS"]).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(losses["TOTAL_LOSS"].item(), x.shape[0])
            if steps % 10 == 0:
                logger.info(
                    f"Epoch[{epoch}] Iteration[{n_iter + 1}/{iters_per_epoch}] "
                    f"Loss: {loss_meter.avg:.3f}, Base Lr: {optimizer.param_groups[0]['lr']:.2e}"
                )

            steps += 1
            for name, value in losses.items():
                writer.add_scalar(f"Loss/{name}", value, steps)

            if steps <= cfg.TRAIN.SCHEDULER.WARMUP_STEPS and scale == scaler.get_scale():
                scheduler["warmup_sched"].step()

        if steps > cfg.TRAIN.SCHEDULER.WARMUP_STEPS:
            scheduler["lr_sched"].step()

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch} done in {elapsed:.0f}s")

        if epoch % cfg.GLOBAL.EVAL_PERIOD == 0 or epoch == cfg.TRAIN.MAX_EPOCHS:
            for name, (query, gallery) in data_manager.test_loader.items():
                rank1, rank5, rank10, mAP = evaluate(model, query, gallery, epoch)
                writer.add_scalar(f"{name} Metrics/mAP", mAP, epoch)
                writer.add_scalar(f"{name} Metrics/r1", rank1, epoch)
                writer.add_scalar(f"{name} Metrics/r5", rank5, epoch)
                writer.add_scalar(f"{name} Metrics/r10", rank10, epoch)

    # All reported results use the last epoch
    torch.save(model.state_dict(), save_dir / "model_last.pth")
    logger.info(f"Saved {save_dir / 'model_last.pth'}")


def eval(cfg):
    data_manager = DataManager(cfg)
    data_manager.summary()

    model = ReID(cfg.MODEL.REID, data_manager.train_loader.dataset.num_classes)
    model.load_checkpoint(cfg.TEST.WEIGHTS)
    model.to("cuda").eval()

    for query, gallery in data_manager.test_loader.values():
        evaluate(model, query, gallery, cfg.TRAIN.MAX_EPOCHS)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = setup(args)
    set_seed(config.GLOBAL.SEED)

    if config.GLOBAL.RUN_MODE == "eval":
        eval(config)
    else:
        if config.GLOBAL.MODE == "debug":
            logger.info("Running in debug mode")
            config.GLOBAL.NUM_WORKERS = 0
            config.GLOBAL.EVAL_PERIOD = 1
        train(config)
