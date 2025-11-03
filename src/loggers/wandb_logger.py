"""Weights & Biases 日志封装。

参考 SOTA 项目的常用实现方式，提供便捷的初始化、指标记录和资源清理接口，
并确保在分布式训练中只由主进程执行 W&B 操作。
"""

from __future__ import annotations

import atexit
import contextlib
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    """检测是否为主进程。"""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


class WandBLogger:
    """W&B 日志管理器。

    Attributes:
        enabled: 是否启用 W&B
        run: wandb.Run 对象（可能为 None）
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        exp_dir: Path,
        *,
        enable: bool,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        resume: str = "allow",
        watch: Optional[str] = None,
        log_checkpoints: bool = True,
        log_images: bool = False,
    ) -> None:
        self.enabled = enable and _is_rank_zero()
        self.exp_dir = Path(exp_dir)
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.tags = list(tags or [])
        self.job_type = job_type
        self.resume = resume
        self.watch = watch
        self.log_checkpoints = log_checkpoints
        self.log_images = log_images
        self.run = None
        self._cfg = cfg
        self._wandb = None
        self._watch_mode = None

        if enable and not _is_rank_zero():
            log.debug("W&B 仅在主进程启用，当前进程跳过初始化。")

        if not self.enabled:
            return

        try:
            import wandb

            settings = dict(dir=str(self.exp_dir / "logs" / "wandb"))
            if run_name is None:
                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"run_{timestamp}"

            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=self.tags,
                job_type=job_type,
                resume=resume,
                config=cfg,
                dir=settings["dir"],
                settings=wandb.Settings(_disable_stats=True),
            )

            log.info("W&B 已初始化: project=%s name=%s", project, run_name)

            if watch:
                wandb.watch_called = False
            self._watch_mode = watch
            self._wandb = wandb
            atexit.register(self.close)
        except ModuleNotFoundError:
            self.enabled = False
            log.warning("未安装 wandb 库，跳过 W&B 日志。请运行 pip install wandb")
        except Exception as exc:  # pragma: no cover - 初始化失败一般不测试
            self.enabled = False
            log.warning("W&B 初始化失败: %s", exc, exc_info=True)

    def watch_model(self, model: torch.nn.Module) -> None:
        """注册模型梯度/参数监控。"""

        if not self.enabled or self._watch_mode is None:
            return
        with contextlib.suppress(Exception):
            self._wandb.watch(model, log=self._watch_mode)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """记录标量指标。"""

        if not self.enabled or self.run is None:
            return
        self.run.log(dict(metrics), step=step)

    def log_image(self, key: str, image: Any, *, step: Optional[int] = None) -> None:
        """上传图像到 W&B。"""

        if not self.enabled or not self.log_images:
            return
        if self.run is None:
            return
        try:
            wandb = self._wandb
            if wandb is None:
                return

            if isinstance(image, Path):
                payload = wandb.Image(str(image))
            elif torch.is_tensor(image):
                payload = wandb.Image(image.detach().cpu().numpy())
            elif isinstance(image, np.ndarray):
                payload = wandb.Image(image)
            else:
                payload = image

            self.run.log({key: payload}, step=step)
        except Exception as exc:
            log.debug("上传图像到 W&B 失败: %s", exc)

    def log_checkpoint(self, ckpt_path: Path, *, name: Optional[str] = None) -> None:
        """上传 checkpoint 文件到 W&B。"""

        if not self.enabled or not self.log_checkpoints or self.run is None:
            return
        try:
            artifact = self._wandb.Artifact(name or ckpt_path.stem, type="model")
            artifact.add_file(str(ckpt_path))
            self.run.log_artifact(artifact)
        except Exception as exc:
            log.debug("上传 checkpoint 到 W&B 失败: %s", exc)

    def set_summary(self, metrics: Mapping[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        for key, value in metrics.items():
            self.run.summary[key] = value

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            if self.run is not None:
                self.run.finish()
                self.run = None
        except Exception as exc:
            log.debug("关闭 W&B 运行失败: %s", exc)


