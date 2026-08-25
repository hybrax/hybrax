"""Self-contained training checkpoint writer."""

from __future__ import annotations

import gzip
import shutil
import time
from pathlib import Path

import optax

from .serialization import save_model, save_opt_state, write_json
from .wrapper import HybridOdeWrapper


def _bundle_prepared_gz(src: Path, dst: Path) -> None:
    if src.suffix == ".gz" or src.name.endswith(".json.gz"):
        shutil.copyfile(src, dst)
        return
    with open(src, "rb") as source, gzip.open(dst, "wb") as destination:
        shutil.copyfileobj(source, destination)


class CheckpointWriter:
    """Writes self-contained ``checkpoints/step_NNNNN/`` directories and
    updates ``latest``.

    Each checkpoint bundles everything needed to resume or reload the run:
    trained params, optimizer state, training-progress metadata, and (when
    available) the run's ``config.json``/``custom.py``/prepared data.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        *,
        prepared_src: Path | None = None,
    ) -> None:
        """Create ``checkpoints_dir`` if needed.

        Args:
            checkpoints_dir: Directory every ``step_NNNNN`` checkpoint and
                ``latest`` are written under.
            prepared_src: Path to the run's prepared-data file, bundled as
                ``prepared.json.gz`` into every checkpoint; omit to skip
                bundling it.
        """
        self._dir = Path(checkpoints_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prepared_src = Path(prepared_src) if prepared_src is not None else None

    def write(
        self,
        *,
        step: int,
        samples_seen: int,
        wrapper: HybridOdeWrapper,
        opt_state: optax.OptState,
        mean_loss: float,
        holdout_loss: float | None,
    ) -> Path:
        """Write one checkpoint directory and point ``latest`` at it.

        Args:
            step: Optimizer step this checkpoint is taken at; names the
                checkpoint directory (``step_{step:05d}``).
            samples_seen: Cumulative training samples processed so far.
            wrapper: Trained wrapper whose params are saved to ``params.eqx``.
            opt_state: Optimizer state saved to ``opt_state.eqx``.
            mean_loss: Training loss at this step, recorded in
                ``train_state.json``.
            holdout_loss: Holdout/validation loss at this step, or ``None``
                when no holdout was evaluated.

        Returns:
            The checkpoint directory that was written.
        """
        d = self._dir / f"step_{step:05d}"
        d.mkdir(parents=True, exist_ok=True)
        save_model(wrapper, d / "params.eqx")
        save_opt_state(opt_state, d / "opt_state.eqx")
        write_json(
            d / "train_state.json",
            {
                "step": int(step),
                "samples_seen": int(samples_seen),
                "mean_loss": float(mean_loss),
                "holdout_loss": (
                    float(holdout_loss) if holdout_loss is not None else None
                ),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            },
        )

        run_dir = self._dir.parent
        for name in ("config.json", "custom.py"):
            source = run_dir / name
            if source.is_file():
                shutil.copyfile(source, d / name)
        if self._prepared_src is not None and self._prepared_src.is_file():
            _bundle_prepared_gz(self._prepared_src, d / "prepared.json.gz")

        self._update_latest(d)
        return d

    def _update_latest(self, step_dir: Path) -> None:
        """Point ``checkpoints/latest`` at the newest step.

        A symlink where the filesystem supports one, a directory holding a COPY
        where it does not. SMB/NAS shares and Windows-backed mounts (WSL drvfs/9p)
        reject ``os.symlink`` outright, and training onto such a share is a normal
        deployment — the alternative is every fold dying with ``PermissionError``
        after the run has already done its work. Readers are unaffected either way:
        ``checkpoints/latest/params.eqx`` resolves in both forms.
        """
        link = self._dir / "latest"
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        try:
            link.symlink_to(step_dir.name)
            return
        except OSError:
            pass
        # Content-only copy. `shutil.copytree` is not usable here: it also replays
        # permissions and mtimes via `copystat`, which those same filesystems reject,
        # so it fails for a second and unrelated reason.
        link.mkdir(parents=True, exist_ok=True)
        for src in sorted(step_dir.rglob("*")):
            dst = link / src.relative_to(step_dir)
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
