"""Public package API tests."""

import os
import subprocess
import sys

import bp_format


def test_import_does_not_load_jax_but_configures_later_import():
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bp_format; "
            "assert 'jax' not in sys.modules; "
            "from bp_format import BioProcess; "
            "assert sys.modules['jax'].config.x64_enabled",
        ],
        check=True,
        env=env,
    )


def test_import_enables_x64_when_jax_is_already_loaded():
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "false"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax; assert not jax.config.x64_enabled; "
            "import bp_format; assert jax.config.x64_enabled",
        ],
        check=True,
        env=env,
    )


def test_version():
    assert hasattr(bp_format, "__version__")
    assert isinstance(bp_format.__version__, str)


def test_all_exports_resolve_and_are_discoverable():
    assert bp_format.__all__
    for name in bp_format.__all__:
        assert getattr(bp_format, name) is not None
        assert name in dir(bp_format)
