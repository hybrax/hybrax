"""Loading a run's ``custom.py`` and reading its config/hooks."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


def load_custom_module(custom_py: str | Path | None) -> ModuleType | None:
    """Import ``custom_py`` as a standalone module named ``hybrax_train_user_custom``.

    Args:
        custom_py: Path to the ``custom.py`` file, or ``None``.

    Returns:
        The imported module, or ``None`` when ``custom_py`` is ``None``.

    Raises:
        FileNotFoundError: If ``custom_py`` does not exist.
        ImportError: If a module spec cannot be built for ``custom_py``.
    """
    if custom_py is None:
        return None

    path = Path(custom_py)
    if not path.exists():
        raise FileNotFoundError(f"custom.py path does not exist: {path}")

    spec = spec_from_file_location("hybrax_train_user_custom", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom module from {path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_config(module: ModuleType | None, config: Any | None) -> dict[str, Any]:
    """Merge a custom module's config with an explicit override, override wins.

    Reads ``module.get_config()`` if defined, else ``module.CONFIG`` if
    defined, else starts empty; then overlays ``config`` on top.

    Args:
        module: Loaded ``custom.py`` module, or ``None`` to skip it.
        config: Explicit config values to overlay, or ``None``.

    Returns:
        The merged config dict.

    Raises:
        TypeError: If ``module.get_config()`` returns a non-dict, non-``None``
            value, or ``module.CONFIG`` is not a dict.
    """
    resolved: dict[str, Any] = {}

    if module is not None:
        if hasattr(module, "get_config"):
            module_config = module.get_config()
            if module_config is not None:
                if not isinstance(module_config, dict):
                    raise TypeError("custom.get_config() must return a dict or None")
                resolved.update(module_config)
        elif hasattr(module, "CONFIG"):
            if not isinstance(module.CONFIG, dict):
                raise TypeError("custom.CONFIG must be a dict")
            resolved.update(module.CONFIG)

    if config is not None:
        resolved.update(dict(config))

    return resolved


def get_hook(module: ModuleType | None, name: str, default: Any) -> Any:
    """Return ``module``'s attribute ``name`` if defined, else ``default``."""
    if module is None:
        return default
    return getattr(module, name, default)


def hook_is_customized(module: ModuleType | None, name: str) -> bool:
    """Whether custom.py defines an attribute named ``name`` (vs. using the default)."""
    return module is not None and hasattr(module, name)


def split_hooks_by_customization(
    module: ModuleType | None, hook_names: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Split ``hook_names`` into (customized, default) buckets for logging."""
    customized = [name for name in hook_names if hook_is_customized(module, name)]
    default = [name for name in hook_names if name not in customized]
    return customized, default
