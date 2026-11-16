"""Loading a run's ``custom.py`` and reading its config/hooks."""

from __future__ import annotations

import ast
import site
import sys
import sysconfig
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

# Module name ``custom.py`` is imported under; pickled custom classes refer to it.
CUSTOM_MODULE_NAME = "hybrax_train_user_custom"


def load_custom_module(custom_py: str | Path | None) -> ModuleType | None:
    """Import ``custom_py`` as a standalone module named ``hybrax_train_user_custom``.

    ``custom.py`` must be self-contained: only it is saved alongside the model in
    each checkpoint. Before running it, :func:`_local_imports` checks its
    ``import`` statements and this raises if one of them loads a file from
    ``custom.py``'s own folder. It guards against that common mistake; it cannot
    see imports made by other packages or built at runtime.

    Args:
        custom_py: Path to the ``custom.py`` file, or ``None``.

    Returns:
        The imported module, or ``None`` when ``custom_py`` is ``None``.

    Raises:
        FileNotFoundError: If ``custom_py`` does not exist.
        ValueError: If ``custom_py`` imports a file from its own folder.
        ImportError: If a module spec cannot be built for ``custom_py``.
    """
    if custom_py is None:
        return None

    path = Path(custom_py)
    if not path.exists():
        raise FileNotFoundError(f"custom.py path does not exist: {path}")

    local = _local_imports(path)
    if local:
        files = ", ".join(local)
        raise ValueError(
            f"custom.py must be self-contained, but {path} imports code from its "
            f"own folder: {files}. Only custom.py is saved alongside the model in "
            "each checkpoint. Therefore, it needs to contain all the helpers etc. it "
            f"uses. Please copy the contents of {files} into custom.py."
        )

    spec = spec_from_file_location(CUSTOM_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom module from {path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _local_imports(custom_py: Path) -> list[str]:
    """Files in ``custom_py``'s folder tree that its ``import`` statements load.

    Reads every ``import`` statement in the source, including ones inside
    functions, and locates each top-level module with :func:`find_spec`, which
    does not run it and also reports where an already imported module came from.
    Modules in the interpreter's standard library or site-packages are exempt, so
    a virtual environment inside the project folder does not count. Returns paths
    relative to the folder, e.g. ``helpers.py`` or ``mypkg/`` for a package.
    """
    # Bytes, so Python applies the same encoding rules as when importing.
    tree = ast.parse(custom_py.read_bytes(), filename=str(custom_py))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.partition(".")[0])

    folder = custom_py.resolve().parent
    environment = [
        Path(p).resolve()
        for p in (
            *(
                sysconfig.get_path(key)
                for key in ("stdlib", "platstdlib", "purelib", "platlib")
            ),
            *site.getsitepackages(),
            site.getusersitepackages(),
        )
    ]

    def is_local(location: Path) -> bool:
        return location.is_relative_to(folder) and not any(
            location.is_relative_to(directory) for directory in environment
        )

    local = []
    for name in sorted(names):
        spec = find_spec(name)
        if spec is None:  # not importable; running custom.py reports that
            continue
        # A namespace package has no file, only search locations.
        locations = (
            [spec.origin]
            if spec.has_location
            else list(spec.submodule_search_locations or ())
        )
        for location in (Path(location).resolve() for location in locations):
            if is_local(location):
                relative = location.relative_to(folder)
                if relative.name == "__init__.py":
                    relative = relative.parent
                is_package = (folder / relative).is_dir()
                local.append(relative.as_posix() + ("/" if is_package else ""))
                break
    return local


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
