"""
计时工具模块
用于记录模型训练和推理过程中各个模块的耗时
参考主流深度学习框架（PyTorch Lightning, MMDetection等）的实现方式
"""

import time
import torch
from collections import OrderedDict
from contextlib import contextmanager


class Timer:
    """
    计时器类，用于记录各个模块的耗时
    支持CPU和GPU计时，自动处理GPU同步
    """
    
    def __init__(self, device=None, sync_cuda=True):
        """
        初始化计时器
        
        Args:
            device: torch.device，用于判断是否需要GPU同步
            sync_cuda: bool，是否在GPU操作后进行同步（确保准确计时）
        """
        self.device = device
        self.sync_cuda = sync_cuda and (device is not None and device.type == 'cuda')
        self.timings = OrderedDict()  # 存储各个模块的耗时列表
        self.counts = OrderedDict()   # 存储各个模块的调用次数
        self.current_timings = {}     # 当前迭代的耗时
        
    def reset(self):
        """重置所有计时数据"""
        self.timings.clear()
        self.counts.clear()
        self.current_timings.clear()
    
    @contextmanager
    def timer(self, name):
        """
        上下文管理器，用于记录代码块的执行时间
        
        使用示例:
            with timer.timer('forward'):
                output = model(input)
        """
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)
        start_time = time.time()
        
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize(self.device)
            elapsed_time = time.time() - start_time
            
            # 记录耗时
            if name not in self.timings:
                self.timings[name] = []
                self.counts[name] = 0
            self.timings[name].append(elapsed_time)
            self.counts[name] += 1
            self.current_timings[name] = elapsed_time
    
    def start(self, name):
        """
        开始计时（手动模式）
        
        Args:
            name: 计时器名称
        """
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)
        self._start_times = getattr(self, '_start_times', {})
        self._start_times[name] = time.time()
    
    def end(self, name):
        """
        结束计时（手动模式）
        
        Args:
            name: 计时器名称
            
        Returns:
            float: 耗时（秒）
        """
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)
        end_time = time.time()
        
        if not hasattr(self, '_start_times') or name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started. Call start('{name}') first.")
        
        elapsed_time = end_time - self._start_times[name]
        
        # 记录耗时
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        self.timings[name].append(elapsed_time)
        self.counts[name] += 1
        self.current_timings[name] = elapsed_time
        
        del self._start_times[name]
        return elapsed_time
    
    def get_current_timings(self):
        """
        获取当前迭代的耗时字典
        
        Returns:
            dict: 当前迭代各个模块的耗时
        """
        return self.current_timings.copy()
    
    def get_average_timings(self, reset=False):
        """
        获取平均耗时
        
        Args:
            reset: bool，是否在获取后重置计时数据
            
        Returns:
            dict: 各个模块的平均耗时
        """
        avg_timings = OrderedDict()
        for name, times in self.timings.items():
            if len(times) > 0:
                avg_timings[name] = sum(times) / len(times)
            else:
                avg_timings[name] = 0.0
        
        if reset:
            self.reset()
        
        return avg_timings
    
    def get_total_timings(self):
        """
        获取总耗时
        
        Returns:
            dict: 各个模块的总耗时
        """
        total_timings = OrderedDict()
        for name, times in self.timings.items():
            total_timings[name] = sum(times)
        return total_timings
    
    def get_summary(self, format_str='{name}: {time:.4f}s'):
        """
        获取耗时摘要字符串
        
        Args:
            format_str: 格式化字符串，{name}和{time}会被替换
            
        Returns:
            str: 格式化的耗时摘要
        """
        avg_timings = self.get_average_timings()
        if not avg_timings:
            return "No timing data available."
        
        lines = []
        for name, avg_time in avg_timings.items():
            count = self.counts.get(name, 0)
            total_time = sum(self.timings.get(name, []))
            lines.append(f"{name}: {avg_time:.4f}s (total: {total_time:.4f}s, count: {count})")
        
        return " | ".join(lines)
    
    def get_summary_current(self):
        """
        获取当前迭代的耗时摘要字符串
        
        Returns:
            str: 格式化的当前迭代耗时摘要
        """
        if not self.current_timings:
            return "No timing data for current iteration."
        
        lines = []
        for name, time in self.current_timings.items():
            lines.append(f"{name}: {time:.4f}s")
        
        return " | ".join(lines)
    
    def format_timings_for_log(self, prefix="Timings"):
        """
        格式化耗时信息用于日志输出
        
        Args:
            prefix: 前缀字符串
            
        Returns:
            str: 格式化的日志字符串
        """
        if not self.current_timings:
            return ""
        
        parts = [f"{prefix}:"]
        for name, time in self.current_timings.items():
            parts.append(f"{name}={time:.4f}s")
        
        return " ".join(parts)

