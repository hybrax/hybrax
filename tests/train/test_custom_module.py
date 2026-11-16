"""custom.py must be self-contained: local imports raise, the environment does not."""

import importlib
import site
import sys

import pytest

from hybrax.train.utils import load_custom_module


def _write_helper(folder, name):
    (folder / f"{name}.py").write_text("VALUE = 1\n")


@pytest.mark.parametrize(
    "source",
    [
        "import {name}\n",
        "from {name} import VALUE\n",
        "def get_custom_config(raw, config):\n    import {name}\n    return raw\n",
    ],
    ids=["import", "from_import", "inside_a_hook"],
)
def test_importing_a_helper_from_its_folder_raises(tmp_path, monkeypatch, source):
    name = f"hybrax_test_helper_{tmp_path.name.replace('-', '_')}"
    _write_helper(tmp_path, name)
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(source.format(name=name))
    # As in a notebook or a script run from this folder.
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match=rf"contents of {name}\.py into custom\.py"):
        load_custom_module(custom_py)


def test_a_local_package_is_named_by_its_folder(tmp_path, monkeypatch):
    package = tmp_path / "hybrax_test_local_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    custom_py = tmp_path / "custom.py"
    custom_py.write_text("import hybrax_test_local_pkg\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="hybrax_test_local_pkg/ into custom.py"):
        load_custom_module(custom_py)


def test_a_rejected_custom_py_never_runs(tmp_path, monkeypatch):
    _write_helper(tmp_path, "hybrax_test_unrun_helper")
    marker = tmp_path / "ran"
    custom_py = tmp_path / "custom.py"
    custom_py.write_text(
        f"open({str(marker)!r}, 'w').close()\nimport hybrax_test_unrun_helper\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError):
        load_custom_module(custom_py)
    assert not marker.exists()


def test_an_already_imported_local_helper_still_raises(tmp_path, monkeypatch):
    _write_helper(tmp_path, "hybrax_test_cached_helper")
    custom_py = tmp_path / "custom.py"
    custom_py.write_text("import hybrax_test_cached_helper\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    # Imported earlier, e.g. in a notebook; removed again after the test.
    monkeypatch.setitem(
        sys.modules,
        "hybrax_test_cached_helper",
        importlib.import_module("hybrax_test_cached_helper"),
    )

    with pytest.raises(ValueError, match="hybrax_test_cached_helper.py"):
        load_custom_module(custom_py)


def test_custom_py_may_import_from_a_venv_inside_its_folder(
    tmp_path, monkeypatch, request
):
    site_packages = tmp_path / ".venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    _write_helper(site_packages, "hybrax_test_installed_pkg")
    custom_py = tmp_path / "custom.py"
    custom_py.write_text("import hybrax_test_installed_pkg\n")
    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_packages)])
    request.addfinalizer(lambda: sys.modules.pop("hybrax_test_installed_pkg", None))

    assert load_custom_module(custom_py).hybrax_test_installed_pkg.VALUE == 1


def test_custom_py_in_a_declared_non_utf8_encoding_loads(tmp_path):
    custom_py = tmp_path / "custom.py"
    source = "# -*- coding: latin-1 -*-\nNAME = 'Müller'\n"
    custom_py.write_bytes(source.encode("latin-1"))

    assert load_custom_module(custom_py).NAME == "Müller"
