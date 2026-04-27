from main_train_vrt import compute_training_eta, format_eta


def test_format_eta_uses_hours_minutes_seconds():
    assert format_eta(3661.2) == "01:01:01"


def test_format_eta_clamps_negative_and_invalid_values():
    assert format_eta(-3.0) == "00:00:00"
    assert format_eta(float("nan")) == "00:00:00"


def test_compute_training_eta_uses_remaining_iters_and_step_time():
    assert compute_training_eta(current_step=25, total_iter=100, seconds_per_iter=2.0) == "00:02:30"


def test_compute_training_eta_returns_zero_when_training_is_complete():
    assert compute_training_eta(current_step=100, total_iter=100, seconds_per_iter=2.0) == "00:00:00"
