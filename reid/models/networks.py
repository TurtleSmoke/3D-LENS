from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.models.vit_pytorch import vit_base_patch16_224_TransReID

# Parameters only used by the training losses.
TRAIN_ONLY_PREFIXES = ("heads.weight", "domain_classifier.")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class EmbeddingHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(feat_dim))
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        self.bottleneck.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, features):
        neck_feat = self.bottleneck(features)[..., 0, 0]
        if not self.training:
            return neck_feat
        return {"cls_outputs": F.linear(neck_feat, self.weight), "features": features[..., 0, 0]}


class ReID(nn.Module):
    """
    ViT-B/16 (TransReID implementation) with a domain token (Sec. 3.3).

    The backbone prepends three learned tokens, [CLS] (identity), domain token (real vs synthetic classification)
    and one extra learned token kept for compatibility with the released checkpoints (`view_token`).
    """

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.base = vit_base_patch16_224_TransReID(
            img_size=cfg.IMG_SIZE,
            camera=0,
            view=0,
            patch_size=cfg.PATCH_SIZE,
            stride_size=cfg.STRIDE_SIZE,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            depth=cfg.DEPTH,
            num_heads=cfg.NUM_HEADS,
            in_chans=3,
        )

        in_planes = 768
        self.heads = EmbeddingHead(in_planes, num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_planes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        self.domain_classifier.apply(weights_init_kaiming)

        # Inputs are in [0, 255], normalized to [-1, 1]
        self.register_buffer("pixel_mean", torch.tensor([127.5] * 3).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor([127.5] * 3).view(1, -1, 1, 1), False)

        if cfg.get("PRETRAINED_PATH"):
            self.base.load_param(torch.load(cfg.PRETRAINED_PATH, map_location="cpu", weights_only=True))
            print(f"Loaded ImageNet weights from {cfg.PRETRAINED_PATH}")

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        global_feat, domain_feat, _ = self.base(x)

        outputs = self.heads(global_feat)
        if self.training:
            outputs["domain_logits"] = self.domain_classifier(domain_feat)
        return outputs

    def load_checkpoint(self, path):
        """Load a 3D-LENS checkpoint."""
        state_dict = torch.load(Path(path), map_location="cpu", weights_only=True)
        # Heads of older internal checkpoints that are not part of the released model
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(("adversarial_classifier.", "view_classifier."))
        }
        if "heads.weight" in state_dict and state_dict["heads.weight"].shape != self.heads.weight.shape:
            state_dict = {k: v for k, v in state_dict.items() if k != "heads.weight"}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith(TRAIN_ONLY_PREFIXES)]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint {path} does not match the model: missing={missing}, unexpected={unexpected}"
            )
        print(f"Loaded checkpoint {path}")
