import os

import torch

from utils.utils_runtime import apply_runtime_cpu_config


def test_apply_runtime_cpu_config_sets_threads_and_compile_env(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.append(("interop", value)))

    tmpdir = tmp_path / "tmp"
    cache_dir = tmp_path / "inductor"
    opt = {
        "runtime": {
            "cpu": {
                "torch_num_threads": 8,
                "torch_num_interop_threads": 4,
                "inductor_compile_threads": 16,
                "omp_num_threads": 8,
                "mkl_num_threads": 8,
                "tmpdir": str(tmpdir),
                "inductor_cache_dir": str(cache_dir),
            }
        }
    }

    summary = apply_runtime_cpu_config(opt)

    assert calls == [("threads", 8), ("interop", 4)]
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "8"
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "16"
    assert os.environ["TMPDIR"] == str(tmpdir)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache_dir)
    assert tmpdir.is_dir()
    assert cache_dir.is_dir()
    assert summary["torch_num_threads"] == 8
    assert summary["torch_num_interop_threads"] == 4
    assert summary["inductor_compile_threads"] == 16
