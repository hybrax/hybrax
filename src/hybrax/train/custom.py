from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any


def load_custom_module(custom_py: str | Path | None) -> ModuleType | None:
    if custom_py is None:
        return None

    path = Path(custom_py)
    if not path.exists():
        raise FileNotFoundError(f"custom.py path does not exist: {path}")

    spec = spec_from_file_location("bp_train_user_custom", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom module from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_config(module: ModuleType | None, config: Any | None) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    if module is not None:
        if hasattr(module, "get_config"):
            module_config = module.get_config()
            if module_config is not None:
                resolved.update(dict(module_config))
        elif hasattr(module, "CONFIG"):
            resolved.update(dict(module.CONFIG))

    if config is not None:
        resolved.update(dict(config))

    return resolved


def get_hook(module: ModuleType | None, name: str, default: Any) -> Any:
    if module is None:
        return default
    return getattr(module, name, default)
