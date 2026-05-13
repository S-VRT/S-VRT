from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_launch_test_script_passes_fusion_debug_overrides_to_main_test():
    launch_script = (REPO_ROOT / "launch_test.sh").read_text(encoding="utf-8")

    for option in (
        "--fusion-debug",
        "--fusion-debug-dir",
        "--fusion-debug-subdir",
        "--fusion-debug-source-view",
        "--fusion-debug-max-batches",
    ):
        assert option in launch_script

    assert "TEST_EXTRA_ARGS" in launch_script
    assert '--fusion_debug' in launch_script
    assert '--fusion_debug_dir "$FUSION_DEBUG_DIR"' in launch_script
    assert '--fusion_debug_subdir "$FUSION_DEBUG_SUBDIR"' in launch_script
    assert '--fusion_debug_source_view "$FUSION_DEBUG_SOURCE_VIEW"' in launch_script
    assert '--fusion_debug_max_batches "$FUSION_DEBUG_MAX_BATCHES"' in launch_script
    assert 'main_test_vrt.py --opt "$RUNTIME_CONFIG" "${TEST_EXTRA_ARGS[@]}"' in launch_script
