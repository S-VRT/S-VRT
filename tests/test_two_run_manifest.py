import json

import pytest

from utils.utils_two_run import (
    build_initial_two_run_state,
    load_two_run_state,
    mark_phase1_completed,
    mark_phase2_started,
    resolve_resume_phase,
    save_two_run_state,
    two_run_state_path,
    update_last_successful_step,
)


def test_build_initial_two_run_state_sets_phase1_as_entrypoint(tmp_path):
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    assert state["current_phase"] == "phase1"
    assert state["phase1_completed"] is False
    assert state["phase2_started"] is False
    assert state["global_step_offset"] == 0


def test_save_and_load_two_run_state_roundtrip(tmp_path):
    state_path = tmp_path / "two_run_state.json"
    state = build_initial_two_run_state(phase1_total_iter=4, phase2_total_iter=6)
    save_two_run_state(state_path, state)
    loaded = load_two_run_state(state_path)
    assert loaded == state


def test_mark_phase1_completed_records_boundary_checkpoints(tmp_path):
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e="models/4000_E.pth")
    assert state["phase1_completed"] is True
    assert state["current_phase"] == "phase2"
    assert state["phase1_final_G"] == "models/4000_G.pth"
    assert state["phase1_final_E"] == "models/4000_E.pth"
    assert state["global_step_offset"] == 4000


def test_resolve_resume_phase_routes_incomplete_phase1_to_phase1_resume():
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    assert resolve_resume_phase(state) == "phase1_resume"


def test_resolve_resume_phase_routes_completed_phase1_to_phase2():
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e=None)
    assert resolve_resume_phase(state) == "phase2_fresh"


def test_resolve_resume_phase_routes_started_phase2_to_phase2_resume():
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e=None)
    mark_phase2_started(state)
    assert resolve_resume_phase(state) == "phase2_resume"


def test_two_run_state_path_uses_task_directory(tmp_path):
    opt = {"path": {"task": str(tmp_path / "experiment")}}
    path = two_run_state_path(opt)
    assert path == tmp_path / "experiment" / "two_run_state.json"


def test_resolve_resume_phase_returns_phase1_fresh_for_none_state():
    assert resolve_resume_phase(None) == "phase1_fresh"


def test_save_two_run_state_overwrites_existing_file_atomically(tmp_path):
    state_path = tmp_path / "two_run_state.json"
    save_two_run_state(state_path, {"a": 1})
    save_two_run_state(state_path, {"a": 2, "b": 3})
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded == {"a": 2, "b": 3}


def test_build_initial_two_run_state_rejects_non_int_total_iter():
    with pytest.raises(ValueError):
        build_initial_two_run_state(phase1_total_iter=4.5, phase2_total_iter=6)
    with pytest.raises(ValueError):
        build_initial_two_run_state(phase1_total_iter=4, phase2_total_iter="6")
    with pytest.raises(ValueError):
        build_initial_two_run_state(phase1_total_iter=True, phase2_total_iter=6)


def test_update_last_successful_step_requires_non_negative_int():
    state = build_initial_two_run_state(phase1_total_iter=4, phase2_total_iter=6)
    update_last_successful_step(state, phase_step=2, global_step=7)
    assert state["last_successful_phase_step"] == 2
    assert state["last_successful_global_step"] == 7
    with pytest.raises(ValueError):
        update_last_successful_step(state, phase_step=-1, global_step=7)
    with pytest.raises(ValueError):
        update_last_successful_step(state, phase_step=1.2, global_step=7)
    with pytest.raises(ValueError):
        update_last_successful_step(state, phase_step=1, global_step=True)
