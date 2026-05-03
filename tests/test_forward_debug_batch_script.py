from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_forward_debug_experiments.sh"


def test_forward_debug_batch_script_targets_requested_experiments():
    text = SCRIPT.read_text(encoding="utf-8")

    for exp_name in (
        "gopro_rawwin9_server_pase_scflow_snapshot",
        "gopro_rawwin9_spynet_pase_residual_dcn4",
        "gopro_tfp4_scflow_mamba_collapsed_dcn4",
        "gopro_tfp4_scflow_mamba_expanded_dcn4",
        "gopro_tfp4_scflow_pase_residual_dcn4",
    ):
        assert f"experiments/{exp_name}" in text


def test_forward_debug_batch_script_self_wraps_with_screen_transcript():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SVRT_FORWARD_DEBUG_INNER" in text
    assert "start_terminal_transcript_wrapper" in text
    assert "script -q -f -e" in text
    assert "screen -S" in text
    assert "screen -dmS" in text
    assert "--detach" in text
    assert "--foreground" in text
    assert "--no-terminal-log" in text


def test_forward_debug_batch_script_materializes_debug_runtime_config():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--artifact-root" in text
    assert "ARTIFACT_ROOT" in text
    assert 'path_cfg["images"] = str(image_root)' in text
    assert 'pathlib.Path(artifact_root).expanduser() / exp.name / run_id / checkpoint_stem' in text
    assert "from main_test_vrt import" not in text
    assert "def strip_json_comments(text):" in text
    assert '"pretrained_netG"] = checkpoint' in text
    assert '"pretrained_netE"] = None' in text
    assert 'forward_debug_${CHECKPOINT_STEM}_${RUN_ID}.json' in text


def test_forward_debug_batch_script_forces_all_test_analysis_dataset():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ANALYSIS_DATASET_SOURCE" in text
    assert '"test_GT_all"' in text
    assert '"test_GT_blurred_all"' in text
    assert '"spike_test_all"' in text
    assert '"meta_info_GoPro_test_all_GT.txt"' in text
    assert 'test_cfg["dataroot_gt"] = str(analysis_dataset / "test_GT_all")' in text


def test_forward_debug_batch_script_limits_test_dataloader_workers_for_parallel_runs():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'TEST_NUM_WORKERS="0"' in text
    assert "--test-num-workers N" in text
    assert "TEST_NUM_WORKERS" in text
    assert 'if [[ ! "$TEST_NUM_WORKERS" =~ ^[0-9]+$ ]]; then' in text
    assert 'test_cfg["dataloader_num_workers"] = int(test_num_workers)' in text
    assert "Test dataloader workers: $TEST_NUM_WORKERS" in text


def test_forward_debug_batch_script_runs_fusion_attribution_directly():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'scripts/analysis/fusion_attribution.py' in text
    assert '--samples "$SAMPLES_FILE"' in text
    assert '--checkpoint "$checkpoint"' in text
    assert '--out "$debug_root"' in text
    assert "--cam-scopes fullframe roi" in text
    assert "docs/analysis/fusion_samples.example.json" in text
    assert "DRY RUN: CUDA_VISIBLE_DEVICES=$assigned_gpu ${debugger_cmd[*]}" in text
    assert "launch_test.sh" not in text


def test_forward_debug_batch_script_runs_full_inference_before_debugger_by_default():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'STAGE="both"' in text
    assert "--stage both|inference|debugger" in text
    assert 'uv run python -u main_test_vrt.py --opt "$runtime_config"' in text
    assert 'CUDA_VISIBLE_DEVICES="$assigned_gpu" "${inference_cmd[@]}"' in text
    assert 'CUDA_VISIBLE_DEVICES="$assigned_gpu" "${debugger_cmd[@]}"' in text
    assert "Full inference output:" in text
    assert "Debugger output:" in text
    assert 'debug_root="$run_root/debugger"' in text


def test_forward_debug_batch_script_maps_old_max_batches_to_max_samples():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--max-samples N" in text
    assert "--fusion-debug-max-batches N" in text
    assert "MAX_SAMPLES" in text
    assert "--max-samples \"$MAX_SAMPLES\"" in text


def test_forward_debug_batch_script_uses_memory_safe_attribution_defaults():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ANALYSIS_NUM_FRAMES="6"' in text
    assert 'ANALYSIS_CROP_SIZE="64"' in text
    assert 'ANALYSIS_TILE_STRIDE="64"' in text
    assert '--analysis-num-frames "$ANALYSIS_NUM_FRAMES"' in text
    assert '--analysis-crop-size "$ANALYSIS_CROP_SIZE"' in text
    assert '--analysis-tile-stride "$ANALYSIS_TILE_STRIDE"' in text


def test_forward_debug_batch_script_parallelizes_across_gpus_by_default():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'GPU_COUNT="auto"' in text
    assert 'WORKERS_PER_GPU="1"' in text
    assert "detect_gpu_count()" in text
    assert "PARALLEL_MODE=true" in text
    assert "--sequential" in text
    assert "run_parallel_experiments()" in text
    assert 'TOTAL_WORKER_SLOTS=$((GPU_COUNT * WORKERS_PER_GPU))' in text
    assert 'assigned_gpu="${GPU_ID_ARRAY[$((slot % GPU_COUNT))]}"' in text
    assert 'run_one_experiment "$exp_dir" "$assigned_gpu"' in text
    assert "wait -n" in text


def test_forward_debug_batch_script_supports_multiple_workers_per_gpu():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--workers-per-gpu N" in text
    assert "WORKERS_PER_GPU" in text
    assert 'if [[ ! "$WORKERS_PER_GPU" =~ ^[0-9]+$ || "$WORKERS_PER_GPU" -lt 1 ]]; then' in text
    assert "Total worker slots: $TOTAL_WORKER_SLOTS" in text
    assert 'assigned_gpu="${GPU_ID_ARRAY[$((idx % GPU_COUNT))]}"' in text
