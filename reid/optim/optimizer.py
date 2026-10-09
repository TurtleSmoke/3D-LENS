# Optimizer adapted from fast-reid (https://github.com/JDAI-CV/fast-reid), by liaoxingyu, itself based on detectron2.
import copy
import itertools
from enum import Enum
from typing import Callable, Iterable, Optional, Type, Union
import re

import torch

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr=None,
    weight_decay=None,
    weight_decay_norm=None,
    bias_lr_factor=1.0,
    heads_lr_factor=1.0,
    weight_decay_bias=None,
    overrides=None,
    freeze_layers=None,
):
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        heads_lr_factor: multiplier of lr for model.head parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
        freeze_layers: layer names for freezing.
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    layer_names_pattern = [re.compile(name) for name in freeze_layers or []]

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params = []
    memo = set()

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            hyperparams.update(overrides.get(module_param_name, {}))
            if module_name.split(".")[0] == "heads" and (heads_lr_factor is not None and heads_lr_factor != 1.0):
                hyperparams["lr"] = hyperparams.get("lr", base_lr) * heads_lr_factor
            name = module_name + "." + module_param_name
            freeze_status = "normal"
            # Search freeze layer names, it must match from beginning, so use `match` not `search`
            for pattern in layer_names_pattern:
                if pattern.match(name) is not None:
                    freeze_status = "freeze"
                    break

            params.append({"freeze_status": freeze_status, "params": [value], **hyperparams})
    return params


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper() -> _GradientClipper:
    """Creates the gradient clipping closure, clipping by L2 norm (max norm 5)."""

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, 5.0, 2.0)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, 5.0)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType("norm")]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert per_param_clipper is None or global_clipper is None, (
        "Not allowed to use both per-parameter clipping and global clipping"
    )

    @torch.no_grad()
    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        optimizer.step(self, closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(optimizer):
    """
    Wraps the optimizer type into a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and clips the gradient of every parameter in its `step` method.
    Args:
        optimizer: type. A subclass of torch.optim.Optimizer
    Return:
        type: a subclass of `optimizer` with gradient clipping included in the `step` method.
    """
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper()
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    return OptimizerWithGradientClip


def build_optimizer(model, cfg):
    optim = cfg.NAME
    lr = cfg.LR

    params = get_default_optimizer_params(
        model,
        base_lr=lr,
        weight_decay=cfg.WEIGHT_DECAY,
        weight_decay_norm=0.0005,
        bias_lr_factor=1.0,
        heads_lr_factor=1.0,
        weight_decay_bias=0.0005,
        freeze_layers=[],
    )

    if optim == "sgd":
        optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(params, momentum=0.9, nesterov=False)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    return optimizer
