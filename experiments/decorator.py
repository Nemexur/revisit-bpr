from typing import Any, Callable
import contextlib
from enum import Enum
from pathlib import Path
import signal
import traceback
from types import DynamicClassAttribute, FrameType

from loguru import logger

from experiments.base import Experiment
from experiments.launcher import Launcher, Simple

_PREEMPT_TO_SAVE = {}


class Status(Enum):
    PREEMPTED = "preempted"
    EXCEPTION = "exception"

    @DynamicClassAttribute
    def value(self) -> Any:
        return self._value_

    def __call__(self, value: Any) -> "Status":
        self._value_ = value
        return self


class Distributed(Experiment):
    def __init__(self, exp: Experiment, launcher: Launcher | None = None) -> None:
        self._exp = exp
        self._launcher = launcher or Simple()

    @property
    def metrics(self) -> dict[str, Any]:
        return self._exp.metrics

    def run(self) -> Any:
        self._launcher.spawn(self._run_local)

    def clean(self) -> None:
        self._exp.clean()

    def interrupt(self) -> None:
        self._exp.interrupt()

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._launcher.init_env(local_rank, world_size)
        try:
            self._exp.run()
        finally:
            self._launcher.clean()


class Preemptible(Experiment):
    def __init__(self, exp: Experiment) -> None:
        self.exit_code: int | None = None
        self._exp = exp

    @property
    def metrics(self) -> dict[str, Any]:
        return self._exp.metrics

    @property
    def _is_preempted(self) -> bool:
        return self.exit_code is not None

    def run(self) -> Any:
        self._trap_signals()
        try:
            self._exp.run()
        except Exception as exc:
            return Status.EXCEPTION(value=(exc, traceback.format_exc()))
        finally:
            self._save_objects()
        if self._is_preempted:
            return Status.PREEMPTED(value=self.exit_code)
        return None

    def clean(self) -> None:
        self._exp.clean()

    def interrupt(self) -> None:
        self._exp.interrupt()

    def _trap_signals(self) -> None:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
            signal.signal(s, self._signal_handler)

    def _signal_handler(self, sig: int, _: FrameType | None) -> None:
        with contextlib.suppress(Exception):  # Ignore all excpetions
            self.interrupt()
        logger.error("experiment: interrupted with signal {}", sig)
        self.exit_code = sig

    def _save_objects(self) -> None:
        for path, saver in _PREEMPT_TO_SAVE.items():
            saver(path)


def preemptible_add_to_save(path: Path, saver: Callable[[Path], None]) -> None:
    _PREEMPT_TO_SAVE[path] = saver
