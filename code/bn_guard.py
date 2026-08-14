"""
bn_guard.py
===========
One implementation of the BatchNorm mode guards, shared by every script that
trains/evaluates the v8 rule zoo (BP / FA / PC / STDP).

WHY THIS EXISTS
---------------
PC_CNN and STDP_CNN used to define `def eval(self): pass`. PC_CNN is an nn.Module,
so that override *shadowed* nn.Module.eval(); STDP_CNN is a plain class, so it never
had one to begin with. Either way `model.eval()` was a silent no-op for those two
rules, and their BatchNorm layers stayed in training mode during evaluation:

  * features were normalised by per-batch statistics of the *evaluation* images at
    whatever resolution was being probed -- a rule-correlated AND resolution-correlated
    confound, in papers whose claims are about rules and resolution;
  * BN running stats were mutated by evaluation, so evaluating changed the model.

The fix is real train()/eval() methods plus explicit mode setting at each phase. These
guards are the check that would have caught it, and they cover BOTH directions:

  assert_bn_eval(model)   -- before feature extraction. Catches "never left training
                             mode", i.e. evaluation contaminated by eval-set batch stats.
  assert_bn_train(model)  -- before a training epoch. Catches the opposite failure: a
                             model that evaluated at a checkpoint and never flipped back,
                             so BN running stats silently freeze for the rest of training.

Scripts that evaluate at intermediate milestones (train -> eval -> train -> ...) need
both, on every cycle.

Usage (Modal): add this module to the image with
    modal.Image....add_local_python_source("bn_guard")
"""

import torch.nn as nn

__all__ = ["iter_bn", "assert_bn_eval", "assert_bn_train"]

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def iter_bn(model):
    """Every BatchNorm layer in `model`, de-duplicated, in discovery order.

    Handles both shapes used in this codebase:
      * nn.Module models (BP_CNN, FA_CNN, PC_CNN) -> walk .modules()
      * plain-class models (STDP_CNN) that merely *hold* nn.Module attributes
        (bn1/bn2/bn3, fc1, ...) -> walk each attribute's .modules()
    """
    if isinstance(model, nn.Module):
        roots = [model]
    else:
        roots = [o for o in vars(model).values() if isinstance(o, nn.Module)]
    found = {}
    for root in roots:
        for m in root.modules():
            if isinstance(m, _BN_TYPES):
                found[id(m)] = m
    return list(found.values())


def _assert_mode(model, want_training, expected_call, why, context):
    bns = iter_bn(model)
    where = f" [{context}]" if context else ""
    if not bns:
        # A vacuous pass is exactly how the original bug hid: if discovery misses the
        # BN layers of the class that had the broken eval(), the check proves nothing.
        raise RuntimeError(
            f"bn_guard{where}: no BatchNorm found in {type(model).__name__}; "
            "the mode check would be vacuous. Fix iter_bn() before trusting this run."
        )
    bad = [i for i, m in enumerate(bns) if m.training is not want_training]
    if bad:
        raise RuntimeError(
            f"bn_guard{where}: {type(model).__name__} has {len(bad)}/{len(bns)} BatchNorm "
            f"layers in {'TRAINING' if not want_training else 'EVAL'} mode "
            f"(indices {bad}) where {'EVAL' if not want_training else 'TRAINING'} was "
            f"required. Did {expected_call} run? {why}"
        )


def assert_bn_eval(model, context=""):
    """Require every BatchNorm to be in eval mode. Call right after model.eval(),
    before extracting features."""
    _assert_mode(
        model, want_training=False, expected_call="model.eval()",
        why="Feature extraction would normalise by per-batch statistics of the "
            "evaluation set instead of the frozen training running stats, and would "
            "mutate the running stats as a side effect.",
        context=context,
    )


def assert_bn_train(model, context=""):
    """Require every BatchNorm to be in training mode. Call right after model.train(),
    before running a training epoch."""
    _assert_mode(
        model, want_training=True, expected_call="model.train()",
        why="Training would run with frozen BatchNorm: running stats would stop "
            "updating for the rest of the run and activations would be normalised by "
            "stale (or, if never trained, identity) statistics.",
        context=context,
    )
