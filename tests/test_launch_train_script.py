from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_with_wrapper_does_not_feed_fifo_to_python_dash():
    launch_script = (REPO_ROOT / "launch_train.sh").read_text(encoding="utf-8")

    assert '<<\'PY\' < "$stdout_pipe"' not in launch_script
    assert '<<\'PY\' < "$stderr_pipe"' not in launch_script


def test_launch_phase_status_messages_use_launch_echo_after_logger_bootstrap():
    launch_script = (REPO_ROOT / "launch_train.sh").read_text(encoding="utf-8")
    after_bootstrap = launch_script.split(
        'LAUNCH_LOG_FILE="$(ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH")"',
        maxsplit=1,
    )[1]

    bare_echo_lines = [
        line.strip()
        for line in after_bootstrap.splitlines()
        if line.strip().startswith("echo ")
    ]

    assert bare_echo_lines == []
