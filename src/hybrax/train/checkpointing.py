"""Self-contained training checkpoint writer."""

from __future__ import annotations

import gzip
import logging
import math
import shutil
import time
from pathlib import Path

import optax

from .postprocessing import DenseProcessExport, export_predictions_csv
from .run_config import CheckpointConfig
from .serialization import save_model, save_opt_state, write_json
from .wrapper import HybridOdeWrapper


logger = logging.getLogger(__name__)
STEP_DIR_PREFIX = "step_"


def _step_dir_name(step: int) -> str:
    """Return the directory name used for a checkpoint's optimizer step."""
    return f"{STEP_DIR_PREFIX}{step:05d}"


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
        keep_best: int | None = None,
        select_by: str | None = None,
    ) -> None:
        """Create ``checkpoints_dir`` if needed.

        Args:
            checkpoints_dir: Directory every ``step_NNNNN`` checkpoint and
                ``latest`` are written under.
            prepared_src: Path to the run's prepared-data file, bundled as
                ``prepared.json.gz`` into every checkpoint; omit to skip
                bundling it.
            keep_best: None keeps all; zero keeps latest only; positive N
                keeps best N plus latest after each successful write.
            select_by: Name of the score supplied to ``write``; required for
                positive N, omitted for zero or None.
        """
        self._dir = Path(checkpoints_dir)
        policy = CheckpointConfig(keep_best=keep_best, select_by=select_by)
        self.check_retention_directory(self._dir, keep_best=policy.keep_best)
        self._keep_best = policy.keep_best
        self._select_by = policy.select_by
        self._retained: dict[int, float | None] = {}
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prepared_src = Path(prepared_src) if prepared_src is not None else None

    @staticmethod
    def check_retention_directory(path: Path, *, keep_best: int | None) -> None:
        """Refuse old checkpoints rather than mix independent training attempts."""
        if keep_best is not None and any(Path(path).glob(f"{STEP_DIR_PREFIX}*")):
            raise ValueError(
                "checkpoint retention requires a fresh checkpoint directory; "
                "use --overwrite or a new output directory"
            )

    def write(
        self,
        *,
        step: int,
        samples_seen: int,
        wrapper: HybridOdeWrapper,
        opt_state: optax.OptState,
        mean_loss: float,
        holdout_loss: float | None,
        holdout_predictions: dict[str, DenseProcessExport] | None = None,
        score: float | None = None,
    ) -> Path:
        """Write one checkpoint directory and point ``latest`` at it.

        With retention enabled, update ``retention.json`` and prune older
        checkpoints after the save and latest update succeed. Keep best N
        plus the new latest, or just latest when ``keep_best=0``.

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
            holdout_predictions: Measurement-grid holdout predictions to write,
                or ``None`` to omit the CSV artifact.
            score: Selection loss for positive keep_best. Lower is better;
                ties keep older checkpoints. Nonfinite scores retain latest
                but cannot enter the best set.

        Returns:
            The checkpoint directory that was written.
        """
        d = self._dir / _step_dir_name(step)
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
                **(
                    {"selection_metric": self._select_by, "selection_score": score}
                    if self._keep_best is not None
                    else {}
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
        if holdout_predictions is not None:
            export_predictions_csv(
                wrapper,
                holdout_predictions,
                d / "holdout_predictions.csv",
            )

        self._update_latest(d)
        if self._keep_best is not None:
            self._prune(step, score)
        return d

    def _prune(self, step: int, score: float | None) -> None:
        """Keep top N finite scores and latest, after the new save succeeds."""
        if self._keep_best and (score is None or not math.isfinite(score)):
            logger.warning(
                "checkpoint step %d has invalid selection score %s", step, score
            )
        self._retained[step] = score
        ranked = sorted(
            (value, saved_step)
            for saved_step, value in self._retained.items()
            if value is not None and math.isfinite(value)
        )[: self._keep_best]
        keep = {saved_step for _, saved_step in ranked} | {step}
        write_json(
            self._dir / "retention.json",
            {
                "metric": self._select_by,
                "keep_best": self._keep_best,
                "latest": _step_dir_name(step),
                "best": [
                    {
                        "step": saved_step,
                        "dir": _step_dir_name(saved_step),
                        "score": value,
                    }
                    for value, saved_step in ranked
                ],
            },
        )
        for saved_step in tuple(self._retained):
            if saved_step not in keep:
                shutil.rmtree(self._dir / _step_dir_name(saved_step))
                del self._retained[saved_step]

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
