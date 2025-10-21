"""
Timing logger utility for performance profiling.

提供两种日志输出方式：
1. 持久化日志文件：记录所有耗时信息供后续分析
2. 终端原地更新：实时显示当前step的耗时信息，不污染主日志
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, TextIO
from collections import defaultdict
import threading


class TimingLogger:
    """
    耗时日志记录器
    
    特性：
    - 分离的日志文件：不污染主训练日志
    - 终端原地更新：使用ANSI转义序列实现实时更新
    - 层级化展示：支持嵌套的耗时信息
    - 统计功能：自动计算平均值、百分比等
    """
    
    def __init__(
        self, 
        log_dir: Path,
        enable_console: bool = True,
        enable_file: bool = True,
        console_update_interval: int = 1,  # 每N个step更新一次终端
        file_flush_interval: int = 10,     # 每N个step刷新一次文件
    ):
        """
        Args:
            log_dir: 日志目录
            enable_console: 是否启用终端输出
            enable_file: 是否启用文件输出
            console_update_interval: 终端更新间隔（步数）
            file_flush_interval: 文件刷新间隔（步数）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.console_update_interval = console_update_interval
        self.file_flush_interval = file_flush_interval
        
        # 文件句柄
        self.file_handle: Optional[TextIO] = None
        if self.enable_file:
            timing_log_path = self.log_dir / f"timing_{int(time.time())}.log"
            self.file_handle = open(timing_log_path, 'w', encoding='utf-8', buffering=1)
            print(f"[TimingLogger] 耗时日志文件: {timing_log_path}")
        
        # 当前step的耗时数据
        self.current_timings: Dict[str, float] = {}
        self.step_count = 0
        
        # 统计数据（用于计算平均值）
        self.timing_history: Dict[str, List[float]] = defaultdict(list)
        
        # 终端显示相关
        self.console_lines = 0  # 上次输出的行数
        self.last_console_update = 0
        
        # 线程锁
        self.lock = threading.Lock()
        
        # ANSI转义序列
        self.CURSOR_UP = '\033[F'  # 光标上移一行
        self.CLEAR_LINE = '\033[K'  # 清除当前行
        self.BOLD = '\033[1m'
        self.RESET = '\033[0m'
        self.CYAN = '\033[36m'
        self.GREEN = '\033[32m'
        self.YELLOW = '\033[33m'
    
    def log_timing(self, name: str, duration_ms: float) -> None:
        """
        记录一个耗时
        
        Args:
            name: 操作名称（支持层级，如 "Spike编码器/尺度0"）
            duration_ms: 耗时（毫秒）
        """
        with self.lock:
            self.current_timings[name] = duration_ms
            self.timing_history[name].append(duration_ms)
    
    def step(self) -> None:
        """
        完成一个step，触发日志输出
        """
        with self.lock:
            self.step_count += 1
            
            # 写入文件
            if self.enable_file and self.file_handle:
                self._write_to_file()
                if self.step_count % self.file_flush_interval == 0:
                    self.file_handle.flush()
            
            # 更新终端显示
            if self.enable_console and self.step_count % self.console_update_interval == 0:
                self._update_console()
            
            # 清空当前step的数据
            self.current_timings.clear()
    
    def _write_to_file(self) -> None:
        """写入文件（详细格式）"""
        if not self.file_handle or not self.current_timings:
            return
        
        self.file_handle.write(f"\n{'='*80}\n")
        self.file_handle.write(f"Step {self.step_count}\n")
        self.file_handle.write(f"{'='*80}\n")
        
        # 计算总耗时
        total_time = sum(v for k, v in self.current_timings.items() if '/' not in k)
        
        # 按层级组织数据
        root_items = [(k, v) for k, v in self.current_timings.items() if '/' not in k]
        
        for name, duration in sorted(root_items, key=lambda x: -x[1]):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            avg = sum(self.timing_history[name]) / len(self.timing_history[name])
            self.file_handle.write(f"{name:40s}: {duration:8.2f}ms ({percentage:5.1f}%) [avg: {avg:8.2f}ms]\n")
            
            # 输出子项
            prefix = name + '/'
            sub_items = [(k, v) for k, v in self.current_timings.items() if k.startswith(prefix)]
            for sub_name, sub_duration in sorted(sub_items, key=lambda x: -x[1]):
                sub_display = sub_name[len(prefix):]
                sub_avg = sum(self.timing_history[sub_name]) / len(self.timing_history[sub_name])
                self.file_handle.write(f"  - {sub_display:36s}: {sub_duration:8.2f}ms [avg: {sub_avg:8.2f}ms]\n")
        
        if total_time > 0:
            self.file_handle.write(f"\n{'Total':-<40s}: {total_time:8.2f}ms\n")
    
    def _update_console(self) -> None:
        """更新终端显示（原地更新，紧凑格式）"""
        if not self.current_timings:
            return
        
        # 清除上次的输出
        if self.console_lines > 0:
            for _ in range(self.console_lines):
                sys.stdout.write(self.CURSOR_UP + self.CLEAR_LINE)
        
        # 准备新的输出
        lines = []
        lines.append(f"{self.CYAN}{self.BOLD}┌─ Timing Profile (Step {self.step_count}) ─────────────────────{self.RESET}")
        
        # 计算总耗时
        total_time = sum(v for k, v in self.current_timings.items() if '/' not in k)
        
        # 只显示顶层项目（紧凑格式）
        root_items = [(k, v) for k, v in self.current_timings.items() if '/' not in k]
        
        for i, (name, duration) in enumerate(sorted(root_items, key=lambda x: -x[1]), 1):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            avg = sum(self.timing_history[name]) / len(self.timing_history[name])
            
            # 绘制进度条
            bar_length = 30
            filled = int(bar_length * percentage / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            lines.append(f"{self.GREEN}│{self.RESET} {name:25s} {bar} {duration:6.1f}ms {percentage:4.1f}%")
        
        if total_time > 0:
            lines.append(f"{self.CYAN}└─ Total: {total_time:6.1f}ms ─────────────────────────────────────{self.RESET}")
        else:
            lines.append(f"{self.CYAN}└────────────────────────────────────────────────────────────{self.RESET}")
        
        # 输出（使用 print 确保立即刷新）
        for line in lines:
            print(line, flush=True)
        
        self.console_lines = len(lines)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取统计摘要
        
        Returns:
            字典，格式: {name: {'avg': xx, 'min': xx, 'max': xx, 'count': xx}}
        """
        summary = {}
        for name, history in self.timing_history.items():
            if history:
                summary[name] = {
                    'avg': sum(history) / len(history),
                    'min': min(history),
                    'max': max(history),
                    'count': len(history),
                }
        return summary
    
    def print_summary(self) -> None:
        """打印统计摘要到终端"""
        summary = self.get_summary()
        if not summary:
            return
        
        print(f"\n{self.BOLD}{'='*80}{self.RESET}")
        print(f"{self.BOLD}Timing Summary (Total {self.step_count} steps){self.RESET}")
        print(f"{self.BOLD}{'='*80}{self.RESET}")
        
        # 只显示顶层项目
        root_items = [(k, v) for k, v in summary.items() if '/' not in k]
        
        for name, stats in sorted(root_items, key=lambda x: -x[1]['avg']):
            print(f"{name:40s}: avg={stats['avg']:7.2f}ms  min={stats['min']:7.2f}ms  max={stats['max']:7.2f}ms  (n={stats['count']})")
    
    def close(self) -> None:
        """关闭日志记录器"""
        with self.lock:
            # 清除终端显示
            if self.enable_console and self.console_lines > 0:
                for _ in range(self.console_lines):
                    sys.stdout.write(self.CURSOR_UP + self.CLEAR_LINE)
                sys.stdout.flush()
            
            # 关闭文件
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局单例（方便在各个模块中使用）
_global_timing_logger: Optional[TimingLogger] = None


def set_global_timing_logger(logger: Optional[TimingLogger]) -> None:
    """设置全局的 TimingLogger 实例"""
    global _global_timing_logger
    _global_timing_logger = logger


def get_global_timing_logger() -> Optional[TimingLogger]:
    """获取全局的 TimingLogger 实例"""
    return _global_timing_logger


def log_timing(name: str, duration_ms: float) -> None:
    """
    便捷函数：记录耗时到全局logger
    
    Args:
        name: 操作名称
        duration_ms: 耗时（毫秒）
    """
    logger = get_global_timing_logger()
    if logger:
        logger.log_timing(name, duration_ms)

