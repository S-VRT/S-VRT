import logging
import os
from pathlib import Path

import torch


def _positive_int(value, key):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{key} must be > 0, got {value}.")
    return value


def _set_env_int(name, value):
    if value is not None:
        os.environ[name] = str(value)


def _set_env_path(name, value):
    if value:
        path = Path(os.path.expanduser(str(value)))
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        return str(path)
    return None


def apply_runtime_cpu_config(opt, logger=None):
    """Apply CPU/threading options before training builds heavy modules."""
    runtime_opt = opt.get('runtime', {}) or {}
    cpu_opt = runtime_opt.get('cpu', {}) or {}
    if not isinstance(cpu_opt, dict):
        raise ValueError("runtime.cpu must be a dict when provided.")

    torch_threads = _positive_int(cpu_opt.get('torch_num_threads'), 'runtime.cpu.torch_num_threads')
    interop_threads = _positive_int(cpu_opt.get('torch_num_interop_threads'), 'runtime.cpu.torch_num_interop_threads')
    inductor_threads = _positive_int(cpu_opt.get('inductor_compile_threads'), 'runtime.cpu.inductor_compile_threads')
    omp_threads = _positive_int(cpu_opt.get('omp_num_threads'), 'runtime.cpu.omp_num_threads')
    mkl_threads = _positive_int(cpu_opt.get('mkl_num_threads'), 'runtime.cpu.mkl_num_threads')
    openblas_threads = _positive_int(
        cpu_opt.get('openblas_num_threads', omp_threads), 'runtime.cpu.openblas_num_threads'
    )
    numexpr_threads = _positive_int(
        cpu_opt.get('numexpr_num_threads', omp_threads), 'runtime.cpu.numexpr_num_threads'
    )

    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass

    _set_env_int('OMP_NUM_THREADS', omp_threads)
    _set_env_int('MKL_NUM_THREADS', mkl_threads)
    _set_env_int('OPENBLAS_NUM_THREADS', openblas_threads)
    _set_env_int('NUMEXPR_NUM_THREADS', numexpr_threads)
    _set_env_int('TORCHINDUCTOR_COMPILE_THREADS', inductor_threads)
    if inductor_threads is not None:
        try:
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = inductor_threads
        except Exception:
            pass
    tmpdir = _set_env_path('TMPDIR', cpu_opt.get('tmpdir'))
    inductor_cache_dir = _set_env_path('TORCHINDUCTOR_CACHE_DIR', cpu_opt.get('inductor_cache_dir'))

    summary = {
        'torch_num_threads': torch_threads,
        'torch_num_interop_threads': interop_threads,
        'inductor_compile_threads': inductor_threads,
        'omp_num_threads': omp_threads,
        'mkl_num_threads': mkl_threads,
        'openblas_num_threads': openblas_threads,
        'numexpr_num_threads': numexpr_threads,
        'tmpdir': tmpdir,
        'inductor_cache_dir': inductor_cache_dir,
    }
    log = logger or logging.getLogger('train')
    if opt.get('rank', 0) == 0:
        log.info('[RUNTIME] cpu config applied: %s', summary)
    return summary
