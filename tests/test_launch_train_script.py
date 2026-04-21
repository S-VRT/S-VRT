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


def test_launch_python_helpers_suppress_utils_option_parse_stdout():
    launch_script = (REPO_ROOT / "launch_train.sh").read_text(encoding="utf-8")

    assert launch_script.count("utils_option.parse(") == launch_script.count(
        "with contextlib.redirect_stdout(io.StringIO()):"
    )


def test_initial_launch_banner_is_logged_in_one_python_call():
    launch_script = (REPO_ROOT / "launch_train.sh").read_text(encoding="utf-8")
    after_bootstrap = launch_script.split(
        'LAUNCH_LOG_FILE="$(ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH")"',
        maxsplit=1,
    )[1]
    banner_block = after_bootstrap.split(
        "# ================================================================================\n# Data Preparation",
        maxsplit=1,
    )[0]

    assert 'launch_echo_lines "train" "launch" "local_single" "info" \\' in banner_block
    assert 'launch_echo "train"' not in banner_block


def test_launch_echo_uses_persistent_logger_daemon():
    launch_script = (REPO_ROOT / "launch_train.sh").read_text(encoding="utf-8")
    launch_echo_body = launch_script.split("launch_echo() {", maxsplit=1)[1].split(
        "\n}\n\nlaunch_emit_record()",
        maxsplit=1,
    )[0]

    assert "start_launch_logger" in launch_script
    assert "LAUNCH_LOG_PIPE" in launch_script
    assert '"$PYTHON_BIN"' not in launch_echo_body
