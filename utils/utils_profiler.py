from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrainProfilerConfig:
    enable: bool
    start_iter: int
    wait: int
    warmup: int
    active: int
    repeat: int
    ranks: tuple[int, ...] | None
    record_shapes: bool
    with_stack: bool
    profile_memory: bool
    trace_dir: Path
    rank: int

    @classmethod
    def from_opt(cls, train_opt: dict[str, Any], experiment_dir, rank: int) -> "TrainProfilerConfig":
        raw = train_opt.get("profiler", {}) or {}
        raw_ranks = raw.get("ranks", [0])
        ranks = None if raw_ranks in (None, "all") else tuple(int(item) for item in raw_ranks)
        trace_root = Path(experiment_dir) / "profiles"
        return cls(
            enable=bool(raw.get("enable", False)),
            start_iter=int(raw.get("start_iter", 0)),
            wait=max(int(raw.get("wait", 1)), 0),
            warmup=max(int(raw.get("warmup", 1)), 0),
            active=max(int(raw.get("active", 2)), 1),
            repeat=max(int(raw.get("repeat", 1)), 1),
            ranks=ranks,
            record_shapes=bool(raw.get("record_shapes", True)),
            with_stack=bool(raw.get("with_stack", False)),
            profile_memory=bool(raw.get("profile_memory", True)),
            trace_dir=trace_root / f"rank{rank}",
            rank=int(rank),
        )

    @property
    def should_profile_rank(self) -> bool:
        return self.enable and (self.ranks is None or self.rank in self.ranks)


class TrainProfiler:
    def __init__(self, cfg: TrainProfilerConfig, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.profiler = None

    def maybe_start(self) -> None:
        if not self.cfg.should_profile_rank:
            return
        self.cfg.trace_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.cfg.wait,
                warmup=self.cfg.warmup,
                active=self.cfg.active,
                repeat=self.cfg.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.cfg.trace_dir)),
            record_shapes=self.cfg.record_shapes,
            profile_memory=self.cfg.profile_memory,
            with_stack=self.cfg.with_stack,
        )
        self.profiler.start()
        if self.logger is not None:
            self.logger.info(f"[profiler] enabled rank={self.cfg.rank} trace_dir={self.cfg.trace_dir}")

    def step(self, current_step: int) -> None:
        if self.profiler is None or current_step < self.cfg.start_iter:
            return
        self.profiler.step()

    def close(self) -> None:
        if self.profiler is None:
            return
        self.profiler.stop()
        if self.logger is not None:
            self.logger.info(f"[profiler] trace_dir={self.cfg.trace_dir}")
        self.profiler = None
