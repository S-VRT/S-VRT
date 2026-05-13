import copy
import json
import os
from datetime import datetime
from pathlib import Path


_FORBIDDEN_PHASE2_TRAIN_VALUES = {
    "G_optimizer_reuse": True,
}

_FORBIDDEN_PHASE2_PATH_KEYS = {
    "pretrained_optimizerG",
}


def deep_merge_dict(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_strict_int(value, label, *, min_value=0):
    if type(value) is not int or value < min_value:
        raise ValueError(f"{label} must be an int >= {min_value}, got {value!r}")


def _require_positive_int(value, label):
    _require_strict_int(value, label, min_value=1)


def validate_two_run_config(opt):
    train = opt.get("train", {})
    two_run = train.get("two_run", {}) or {}
    if not two_run.get("enable", False):
        return

    for phase_name in ("phase1", "phase2"):
        phase_cfg = two_run.get(phase_name)
        if not isinstance(phase_cfg, dict):
            raise ValueError(f"train.two_run.{phase_name} must be a dict")
        _require_positive_int(phase_cfg.get("total_iter"), f"train.two_run.{phase_name}.total_iter")

    phase2_cfg = two_run["phase2"]
    for key, forbidden_value in _FORBIDDEN_PHASE2_TRAIN_VALUES.items():
        if phase2_cfg.get(key, False) == forbidden_value:
            raise ValueError(f"train.two_run.phase2.{key} must not be {forbidden_value!r}")

    phase2_path = phase2_cfg.get("path", {}) or {}
    for key in _FORBIDDEN_PHASE2_PATH_KEYS:
        if key in phase2_path:
            raise ValueError(f"train.two_run.phase2.path.{key} is runtime-owned and must not be set")


def _apply_phase_train_overrides(base_opt, phase_override):
    phase_opt = copy.deepcopy(base_opt)
    train_override = copy.deepcopy(phase_override)
    nested_path_override = train_override.pop("path", None)
    phase_opt["train"] = deep_merge_dict(phase_opt.get("train", {}), train_override)
    if nested_path_override:
        phase_opt["path"] = deep_merge_dict(phase_opt.get("path", {}), nested_path_override)
    return phase_opt


def resolve_two_run_phase_opts(opt):
    validate_two_run_config(opt)
    train = opt.get("train", {})
    two_run = train.get("two_run", {}) or {}
    if not two_run.get("enable", False):
        return None, None

    phase1_opt = _apply_phase_train_overrides(opt, two_run["phase1"])
    phase2_opt = _apply_phase_train_overrides(opt, two_run["phase2"])

    phase1_opt["train"]["two_run"] = copy.deepcopy(two_run)
    phase2_opt["train"]["two_run"] = copy.deepcopy(two_run)
    return phase1_opt, phase2_opt


def dump_resolved_two_run_opts(base_opt, phase1_opt, phase2_opt):
    options_dir = Path(base_opt["path"]["options"])
    options_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "base": options_dir / f"{stamp}_base.json",
        "phase1": options_dir / f"{stamp}_phase1_resolved.json",
        "phase2": options_dir / f"{stamp}_phase2_resolved.json",
    }
    payloads = {
        "base": base_opt,
        "phase1": phase1_opt,
        "phase2": phase2_opt,
    }
    for key, dump_path in paths.items():
        dump_path.write_text(json.dumps(payloads[key], indent=2), encoding="utf-8")
    return paths


def two_run_state_path(opt):
    return Path(opt["path"]["task"]) / "two_run_state.json"


def build_initial_two_run_state(*, phase1_total_iter, phase2_total_iter):
    _require_positive_int(phase1_total_iter, "phase1_total_iter")
    _require_positive_int(phase2_total_iter, "phase2_total_iter")
    return {
        "two_run_enabled": True,
        "current_phase": "phase1",
        "phase1_total_iter": phase1_total_iter,
        "phase2_total_iter": phase2_total_iter,
        "phase1_completed": False,
        "phase1_final_G": None,
        "phase1_final_E": None,
        "phase2_started": False,
        "global_step_offset": 0,
        "last_successful_phase_step": 0,
        "last_successful_global_step": 0,
    }


def load_two_run_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_two_run_state(state_path, state):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f".{state_path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp_path, state_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def mark_phase1_completed(state, *, phase1_final_g, phase1_final_e):
    state["phase1_completed"] = True
    state["current_phase"] = "phase2"
    state["phase1_final_G"] = phase1_final_g
    state["phase1_final_E"] = phase1_final_e
    state["global_step_offset"] = state["phase1_total_iter"]
    state["last_successful_phase_step"] = 0
    state["last_successful_global_step"] = state["global_step_offset"]


def mark_phase2_started(state):
    state["phase2_started"] = True
    state["current_phase"] = "phase2"


def update_last_successful_step(state, *, phase_step, global_step):
    _require_strict_int(phase_step, "phase_step", min_value=0)
    _require_strict_int(global_step, "global_step", min_value=0)
    state["last_successful_phase_step"] = phase_step
    state["last_successful_global_step"] = global_step


def resolve_resume_phase(state):
    if state is None:
        return "phase1_fresh"
    if not state.get("phase1_completed", False):
        return "phase1_resume"
    if not state.get("phase2_started", False):
        return "phase2_fresh"
    return "phase2_resume"
