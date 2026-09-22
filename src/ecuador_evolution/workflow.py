from __future__ import annotations

import importlib

from .config import Settings


def run_experiment(action: str, settings: Settings):
    try:
        module = importlib.import_module("ecuador_evolution.experiment")
    except ModuleNotFoundError as error:
        if error.name != "ecuador_evolution.experiment":
            raise
        raise RuntimeError(
            f"'{action}' is implemented on an experiment branch/worktree; main contains shared infrastructure only"
        ) from error
    function = getattr(module, action, None)
    if function is None:
        raise RuntimeError(f"Experiment module does not implement {action}()")
    return function(settings)
