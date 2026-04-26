"""Configurable torch.profiler helper for training."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.profiler import ProfilerActivity, schedule


@dataclass(frozen=True)
class TrainProfilerConfig:
    enable: bool = False
    start_iter: int = 0
    wait: int = 1
    warmup: int = 1
    active: int = 2
    repeat: int = 1
    ranks: Optional[List[int]] = None  # None / "all" means all ranks
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False
    rank: int = 0
    experiment_dir: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_opt(
        cls,
        train_opt: dict,
        experiment_dir: Union[str, Path],
        rank: int,
    ) -> "TrainProfilerConfig":
        pcfg = train_opt.get("profiler", {})
        enable = bool(pcfg.get("enable", False))

        ranks_raw = pcfg.get("ranks", None)
        if ranks_raw is None or ranks_raw == "all":
            ranks = None  # all ranks
        else:
            ranks = list(ranks_raw)

        return cls(
            enable=enable,
            start_iter=int(pcfg.get("start_iter", 0)),
            wait=max(0, int(pcfg.get("wait", 1))),
            warmup=max(0, int(pcfg.get("warmup", 1))),
            active=max(1, int(pcfg.get("active", 2))),
            repeat=max(1, int(pcfg.get("repeat", 1))),
            ranks=ranks,
            record_shapes=bool(pcfg.get("record_shapes", False)),
            with_stack=bool(pcfg.get("with_stack", False)),
            profile_memory=bool(pcfg.get("profile_memory", False)),
            rank=rank,
            experiment_dir=Path(experiment_dir),
        )

    @property
    def should_profile_rank(self) -> bool:
        if not self.enable:
            return False
        if self.ranks is None:
            return True
        return self.rank in self.ranks

    @property
    def trace_dir(self) -> Path:
        return self.experiment_dir / "profiles" / f"rank{self.rank}"


class TrainProfiler:
    """Thin wrapper around torch.profiler.profile for training loops."""

    def __init__(
        self,
        cfg: TrainProfilerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self._profiler: Optional[torch.profiler.profile] = None
        self._active = False

    def maybe_start(self) -> None:
        if not self.cfg.enable or not self.cfg.should_profile_rank:
            return

        self.cfg.trace_dir.mkdir(parents=True, exist_ok=True)

        sched = schedule(
            wait=self.cfg.wait,
            warmup=self.cfg.warmup,
            active=self.cfg.active,
            repeat=self.cfg.repeat,
        )

        self._profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.cfg.trace_dir)
            ),
            record_shapes=self.cfg.record_shapes,
            with_stack=self.cfg.with_stack,
            profile_memory=self.cfg.profile_memory,
        )
        self._profiler.__enter__()
        self._active = True

        if self.logger is not None:
            self.logger.info(
                "[PROFILER] torch.profiler enabled for rank %d, "
                "start_iter=%d, trace_dir=%s",
                self.cfg.rank,
                self.cfg.start_iter,
                self.cfg.trace_dir,
            )

    def step(self, current_step: int) -> None:
        if not self._active or self._profiler is None:
            return
        if current_step >= self.cfg.start_iter:
            self._profiler.step()

    def close(self) -> None:
        if not self._active or self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        self._active = False
        if self.logger is not None:
            self.logger.info(
                "[PROFILER] Profiler closed. Traces written to %s",
                self.cfg.trace_dir,
            )
