#!/usr/bin/env bash
set -euo pipefail

# Batch full test-set inference plus fusion attribution debug artifacts for selected experiments.
# The outer invocation wraps itself in screen+script by default, matching launch_train.sh.

ORIGINAL_ARGS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_EXPERIMENTS=(
    "experiments/gopro_rawwin9_server_pase_scflow_snapshot"
    "experiments/gopro_rawwin9_spynet_pase_residual_dcn4"
    "experiments/gopro_tfp4_scflow_mamba_collapsed_dcn4"
    "experiments/gopro_tfp4_scflow_mamba_expanded_dcn4"
    "experiments/gopro_tfp4_scflow_pase_residual_dcn4"
)

GPU_COUNT="auto"
GPU_LIST=""
WORKERS_PER_GPU="1"
CHECKPOINT_STEM="10000_G"
SAMPLES_FILE="docs/analysis/fusion_samples.example.json"
ANALYSIS_DATASET_SOURCE="experiments/gopro_tfp4_scflow_mamba_collapsed_dcn4/analysis_dataset"
TEST_NUM_WORKERS="0"
MAX_SAMPLES=""
CAM_METHOD="gradcam"
ANALYSIS_NUM_FRAMES="6"
ANALYSIS_CROP_SIZE="64"
ANALYSIS_TILE_STRIDE="64"
ARTIFACT_ROOT=""
STAGE="both"
TERMINAL_LOG=true
TERMINAL_MODE="attach"
CONTINUE_ON_ERROR=false
PARALLEL_MODE=true
DRY_RUN=false
RUN_ID="${SVRT_FORWARD_DEBUG_RUN_ID:-$(date +%y%m%d_%H%M%S)}"
BATCH_LOG_DIR="experiments/forward_debug_batch"
EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")

usage() {
    cat <<EOF
Usage: $0 [GPU_COUNT] [options]

Runs full test-set inference and/or the offline fusion attribution debugger for the selected experiments.

Arguments:
  GPU_COUNT                         Number of local GPUs to use as worker slots (default: auto)

Options:
  --gpus 0,1,2                      Comma-separated GPU ids; one experiment runs per GPU
  --workers-per-gpu N               Concurrent experiment workers per GPU (default: $WORKERS_PER_GPU)
  --checkpoint-stem NAME            Checkpoint stem under each models dir (default: $CHECKPOINT_STEM)
                                     Example: 10000_G, 10001_G_merged, 10000_E
  --samples PATH                     Sample list for attribution (default: $SAMPLES_FILE)
  --analysis-dataset-source PATH     Dataset root containing test_GT_all, test_GT_blurred_all,
                                     spike_test_all, and meta_info_GoPro_test_all_GT.txt
                                     (default: $ANALYSIS_DATASET_SOURCE)
  --test-num-workers N               DataLoader workers per inference/debugger process (default: $TEST_NUM_WORKERS)
  --max-samples N                    Maximum attribution samples per experiment
  --fusion-debug-max-batches N       Deprecated alias for --max-samples
  --cam-method NAME                  CAM method: gradcam, hirescam, fallback (default: $CAM_METHOD)
  --analysis-num-frames N            Temporal window for attribution (default: $ANALYSIS_NUM_FRAMES)
  --analysis-crop-size N             Spatial tile size for full-frame stitching (default: $ANALYSIS_CROP_SIZE)
  --analysis-tile-stride N           Spatial tile stride (default: $ANALYSIS_TILE_STRIDE)
  --artifact-root PATH               Root for heavy debug artifacts. When set, artifacts go to:
                                     PATH/<experiment>/<run_id>/<checkpoint_stem>/{inference,debugger}/
  --stage both|inference|debugger    Which stage to run (default: $STAGE)
  --experiment PATH                  Add one experiment path; may be repeated. Replaces the default list on first use.
  --parallel                         Run experiments concurrently, one per GPU (default)
  --sequential                       Run experiments one after another
  --continue-on-error                Continue with later experiments if one forward run fails
  --dry-run                          Generate runtime configs and print commands without running inference
  --attach                           Run inside screen and attach immediately (default)
  --detach                           Run inside a detached screen session and return
  --foreground                       Run in current terminal, still recording terminal_*.log
  --no-terminal-log                  Disable screen/script terminal transcript wrapping
  --help | -h                        Show this message

Examples:
  $0 --detach
  $0 4 --gpus 0,1,2,3 --detach
  $0 --artifact-root /root/autodl-tmp/svrt_forward_debug --detach
  $0 --checkpoint-stem 10001_G_merged --samples docs/analysis/fusion_samples.example.json --detach
  screen -r svrt_forward_debug_batch_<timestamp>
EOF
}

custom_experiments=false
while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        --gpus=*)
            GPU_LIST="${arg#*=}"
            shift
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --workers-per-gpu=*)
            WORKERS_PER_GPU="${arg#*=}"
            shift
            ;;
        --workers-per-gpu)
            WORKERS_PER_GPU="$2"
            shift 2
            ;;
        --checkpoint-stem=*)
            CHECKPOINT_STEM="${arg#*=}"
            shift
            ;;
        --checkpoint-stem)
            CHECKPOINT_STEM="$2"
            shift 2
            ;;
        --samples=*)
            SAMPLES_FILE="${arg#*=}"
            shift
            ;;
        --samples)
            SAMPLES_FILE="$2"
            shift 2
            ;;
        --analysis-dataset-source=*)
            ANALYSIS_DATASET_SOURCE="${arg#*=}"
            shift
            ;;
        --analysis-dataset-source)
            ANALYSIS_DATASET_SOURCE="$2"
            shift 2
            ;;
        --test-num-workers=*)
            TEST_NUM_WORKERS="${arg#*=}"
            shift
            ;;
        --test-num-workers)
            TEST_NUM_WORKERS="$2"
            shift 2
            ;;
        --max-samples=*)
            MAX_SAMPLES="${arg#*=}"
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --fusion-debug-max-batches=*)
            MAX_SAMPLES="${arg#*=}"
            shift
            ;;
        --fusion-debug-max-batches)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --fusion-debug-source-view=*|--fusion-debug-subdir=*)
            echo "Warning: $arg is ignored; forward debug now uses full-frame attribution outputs." >&2
            shift
            ;;
        --fusion-debug-source-view|--fusion-debug-subdir)
            echo "Warning: $arg $2 is ignored; forward debug now uses full-frame attribution outputs." >&2
            shift 2
            ;;
        --cam-method=*)
            CAM_METHOD="${arg#*=}"
            shift
            ;;
        --cam-method)
            CAM_METHOD="$2"
            shift 2
            ;;
        --analysis-num-frames=*)
            ANALYSIS_NUM_FRAMES="${arg#*=}"
            shift
            ;;
        --analysis-num-frames)
            ANALYSIS_NUM_FRAMES="$2"
            shift 2
            ;;
        --analysis-crop-size=*)
            ANALYSIS_CROP_SIZE="${arg#*=}"
            shift
            ;;
        --analysis-crop-size)
            ANALYSIS_CROP_SIZE="$2"
            shift 2
            ;;
        --analysis-tile-stride=*)
            ANALYSIS_TILE_STRIDE="${arg#*=}"
            shift
            ;;
        --analysis-tile-stride)
            ANALYSIS_TILE_STRIDE="$2"
            shift 2
            ;;
        --artifact-root=*)
            ARTIFACT_ROOT="${arg#*=}"
            shift
            ;;
        --artifact-root)
            ARTIFACT_ROOT="$2"
            shift 2
            ;;
        --stage=*)
            STAGE="${arg#*=}"
            shift
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --experiment=*)
            if [[ "$custom_experiments" == false ]]; then
                EXPERIMENTS=()
                custom_experiments=true
            fi
            EXPERIMENTS+=("${arg#*=}")
            shift
            ;;
        --experiment)
            if [[ "$custom_experiments" == false ]]; then
                EXPERIMENTS=()
                custom_experiments=true
            fi
            EXPERIMENTS+=("$2")
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --sequential)
            PARALLEL_MODE=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --attach)
            TERMINAL_MODE="attach"
            shift
            ;;
        --detach)
            TERMINAL_MODE="detach"
            shift
            ;;
        --foreground)
            TERMINAL_MODE="foreground"
            shift
            ;;
        --no-terminal-log)
            TERMINAL_LOG=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                GPU_COUNT="$arg"
                shift
            else
                echo "Unknown argument: $arg" >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
done

if [[ ! -f "$SAMPLES_FILE" ]]; then
    echo "Error: samples file not found: $SAMPLES_FILE" >&2
    exit 1
fi

if [[ ! -d "$ANALYSIS_DATASET_SOURCE" ]]; then
    echo "Error: analysis dataset source not found: $ANALYSIS_DATASET_SOURCE" >&2
    exit 1
fi

if [[ ! "$TEST_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Error: --test-num-workers must be a non-negative integer." >&2
    exit 1
fi

for required_analysis_path in \
    "$ANALYSIS_DATASET_SOURCE/test_GT_all" \
    "$ANALYSIS_DATASET_SOURCE/test_GT_blurred_all" \
    "$ANALYSIS_DATASET_SOURCE/spike_test_all" \
    "$ANALYSIS_DATASET_SOURCE/meta_info_GoPro_test_all_GT.txt"; do
    if [[ ! -e "$required_analysis_path" ]]; then
        echo "Error: analysis dataset source is missing: $required_analysis_path" >&2
        exit 1
    fi
done

if [[ -n "$MAX_SAMPLES" && ( ! "$MAX_SAMPLES" =~ ^[0-9]+$ || "$MAX_SAMPLES" -lt 1 ) ]]; then
    echo "Error: --max-samples must be a positive integer." >&2
    exit 1
fi

if [[ "$CAM_METHOD" != "gradcam" && "$CAM_METHOD" != "hirescam" && "$CAM_METHOD" != "fallback" ]]; then
    echo "Error: --cam-method must be gradcam, hirescam, or fallback." >&2
    exit 1
fi

if [[ "$STAGE" != "both" && "$STAGE" != "inference" && "$STAGE" != "debugger" ]]; then
    echo "Error: --stage must be both, inference, or debugger." >&2
    exit 1
fi

if [[ ! "$ANALYSIS_NUM_FRAMES" =~ ^[0-9]+$ || "$ANALYSIS_NUM_FRAMES" -lt 1 ]]; then
    echo "Error: --analysis-num-frames must be a positive integer." >&2
    exit 1
fi

if [[ ! "$ANALYSIS_CROP_SIZE" =~ ^[0-9]+$ || "$ANALYSIS_CROP_SIZE" -lt 1 ]]; then
    echo "Error: --analysis-crop-size must be a positive integer." >&2
    exit 1
fi

if [[ -n "$ANALYSIS_TILE_STRIDE" && ( ! "$ANALYSIS_TILE_STRIDE" =~ ^[0-9]+$ || "$ANALYSIS_TILE_STRIDE" -lt 1 ) ]]; then
    echo "Error: --analysis-tile-stride must be a positive integer." >&2
    exit 1
fi

if [[ ! "$WORKERS_PER_GPU" =~ ^[0-9]+$ || "$WORKERS_PER_GPU" -lt 1 ]]; then
    echo "Error: --workers-per-gpu must be a positive integer." >&2
    exit 1
fi

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
        if [[ "$count" -gt 0 ]]; then
            echo "$count"
            return 0
        fi
    fi
    local count
    count="$(uv run python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || true)"
    if [[ -n "$count" && "$count" -gt 0 ]]; then
        echo "$count"
        return 0
    fi
    echo 1
}

if [[ -n "$GPU_LIST" ]]; then
    GPU_LIST="$(echo "$GPU_LIST" | tr -d '[:space:]')"
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
    GPU_COUNT="${#GPU_ID_ARRAY[@]}"
elif [[ "$GPU_COUNT" == "auto" ]]; then
    GPU_COUNT="$(detect_gpu_count)"
    if [[ "$GPU_COUNT" -le 1 ]]; then
        GPU_LIST="0"
    else
        GPU_LIST="$(seq -s, 0 $((GPU_COUNT - 1)))"
    fi
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
elif [[ "$GPU_COUNT" -eq 1 ]]; then
    GPU_LIST="0"
    GPU_ID_ARRAY=("0")
else
    GPU_LIST="$(seq -s, 0 $((GPU_COUNT - 1)))"
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
fi

TOTAL_WORKER_SLOTS=$((GPU_COUNT * WORKERS_PER_GPU))

quote_shell_word() {
    printf '%q' "$1"
}

build_reentry_command() {
    local command_text arg
    command_text="cd $(quote_shell_word "$REPO_ROOT") && SVRT_FORWARD_DEBUG_INNER=1"
    if [[ -n "${SVRT_TERMINAL_LOG:-}" ]]; then
        command_text+=" SVRT_TERMINAL_LOG=$(quote_shell_word "$SVRT_TERMINAL_LOG")"
    fi
    command_text+=" exec bash $(quote_shell_word "$0")"
    for arg in "$@"; do
        command_text+=" $(quote_shell_word "$arg")"
    done
    printf '%s' "$command_text"
}

start_terminal_transcript_wrapper() {
    local log_dir="$1"
    shift

    local timestamp session_name terminal_log info_file inner_command script_command
    timestamp="$(date +%y%m%d_%H%M%S)"
    session_name="svrt_forward_debug_batch_${timestamp}"
    terminal_log="${log_dir}/terminal_${timestamp}.log"
    info_file="${log_dir}/screen_${timestamp}.info"

    mkdir -p "$log_dir"

    if ! command -v script >/dev/null 2>&1; then
        echo "Warning: script not found; terminal transcript logging is disabled." >&2
        return 0
    fi

    export SVRT_TERMINAL_LOG="$terminal_log"
    inner_command="$(build_reentry_command "$@")"
    script_command="script -q -f -e -c $(quote_shell_word "$inner_command") $(quote_shell_word "$terminal_log")"

    {
        echo "screen_session=$session_name"
        echo "terminal_log=$terminal_log"
        echo "launch_mode=$TERMINAL_MODE"
        echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "reattach_command=screen -r $session_name"
        echo "tail_command=tail -f $terminal_log"
    } > "$info_file"

    if [[ "$TERMINAL_MODE" == "foreground" ]]; then
        echo "Terminal transcript: $terminal_log"
        exec script -q -f -e -c "$inner_command" "$terminal_log"
    fi

    if ! command -v screen >/dev/null 2>&1; then
        echo "Warning: screen not found; falling back to foreground script logging." >&2
        echo "Terminal transcript: $terminal_log"
        exec script -q -f -e -c "$inner_command" "$terminal_log"
    fi

    echo "Screen session: $session_name"
    echo "Terminal transcript: $terminal_log"
    echo "Session info: $info_file"

    if [[ "$TERMINAL_MODE" == "detach" || ! -t 0 ]]; then
        screen -dmS "$session_name" bash -lc "$script_command"
        echo "Detached. Reattach with: screen -r $session_name"
        exit 0
    fi

    exec screen -S "$session_name" bash -lc "$script_command"
}

if [[ "$TERMINAL_LOG" == true && "${SVRT_FORWARD_DEBUG_INNER:-0}" != "1" ]]; then
    start_terminal_transcript_wrapper "$BATCH_LOG_DIR" "${ORIGINAL_ARGS[@]}"
fi

find_base_config() {
    local exp_dir="$1"
    local infer_cfg
    infer_cfg="$exp_dir/options/infer_10000_G_all_test.json"
    if [[ -f "$infer_cfg" ]]; then
        printf '%s\n' "$infer_cfg"
        return 0
    fi
    find "$exp_dir/options" -maxdepth 1 -type f -name 'vrt_config*.json' | sort | tail -n 1
}

materialize_runtime_config() {
    local exp_dir="$1"
    local base_config="$2"
    local checkpoint="$3"
    local output_config="$4"

    python - "$base_config" "$output_config" "$exp_dir" "$checkpoint" "$RUN_ID" "$ARTIFACT_ROOT" "$ANALYSIS_DATASET_SOURCE" "$TEST_NUM_WORKERS" <<'PY'
import json
import pathlib
import sys

base_config, output_config, exp_dir, checkpoint, run_id, artifact_root, analysis_dataset_source, test_num_workers = sys.argv[1:9]
exp = pathlib.Path(exp_dir)
checkpoint_stem = pathlib.Path(checkpoint).stem
analysis_dataset = pathlib.Path(analysis_dataset_source)
if artifact_root:
    image_root = pathlib.Path(artifact_root).expanduser() / exp.name / run_id / checkpoint_stem / "inference"
else:
    image_root = exp / "images" / "forward_debug" / run_id / checkpoint_stem / "inference"

def strip_json_comments(text):
    result = []
    i = 0
    in_string = False
    string_char = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            result.append(ch)
            if ch == "\\" and i + 1 < len(text):
                result.append(text[i + 1])
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                i += 2
                while i < len(text) and text[i] not in "\n\r":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)

with open(base_config, "r", encoding="utf-8") as f:
    cfg = json.loads(strip_json_comments(f.read()))

cfg["is_train"] = False
cfg["task"] = f"{exp.name}_forward_debug_{pathlib.Path(checkpoint).stem}_{run_id}"

path_cfg = cfg.setdefault("path", {})
path_cfg["root"] = "experiments"
path_cfg["pretrained_netG"] = checkpoint
path_cfg["pretrained_netE"] = None
path_cfg["task"] = str(exp)
path_cfg["log"] = str(exp)
path_cfg["options"] = str(exp / "options")
path_cfg["models"] = str(exp / "models")
path_cfg["images"] = str(image_root)
path_cfg["tensorboard"] = str(exp / "tensorboard")
path_cfg["pretrained_optimizerG"] = None

test_cfg = cfg.setdefault("datasets", {}).setdefault("test", {})
test_cfg["phase"] = "test"
test_cfg["dataroot_gt"] = str(analysis_dataset / "test_GT_all")
test_cfg["dataroot_lq"] = str(analysis_dataset / "test_GT_blurred_all")
test_cfg["dataroot_spike"] = str(analysis_dataset / "spike_test_all")
test_cfg["meta_info_file"] = str(analysis_dataset / "meta_info_GoPro_test_all_GT.txt")
test_cfg["dataloader_shuffle"] = False
test_cfg["dataloader_batch_size"] = 1
test_cfg["dataloader_num_workers"] = int(test_num_workers)
test_cfg["cache_data"] = False

val_cfg = cfg.setdefault("val", {})
val_cfg.setdefault("save_img", True)

out = pathlib.Path(output_config)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(out)
PY
}

run_one_experiment() {
    local exp_dir="$1"
    local assigned_gpu="${2:-0}"
    local base_config checkpoint runtime_config run_root inference_result_root debug_root cmd_exit
    local -a inference_cmd debugger_cmd

    if [[ ! -d "$exp_dir" ]]; then
        echo "[ERROR] Missing experiment directory: $exp_dir" >&2
        return 1
    fi

    base_config="$(find_base_config "$exp_dir")"
    if [[ -z "$base_config" || ! -f "$base_config" ]]; then
        echo "[ERROR] No base config found under $exp_dir/options" >&2
        return 1
    fi

    checkpoint="$exp_dir/models/${CHECKPOINT_STEM}.pth"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[ERROR] Missing checkpoint: $checkpoint" >&2
        return 1
    fi

    runtime_config="$exp_dir/options/forward_debug_${CHECKPOINT_STEM}_${RUN_ID}.json"
    materialize_runtime_config "$exp_dir" "$base_config" "$checkpoint" "$runtime_config"
    if [[ -n "$ARTIFACT_ROOT" ]]; then
        run_root="$ARTIFACT_ROOT/$(basename "$exp_dir")/$RUN_ID/$CHECKPOINT_STEM"
    else
        run_root="$exp_dir/images/forward_debug/$RUN_ID/$CHECKPOINT_STEM"
    fi
    inference_result_root="results/$(basename "$exp_dir")_forward_debug_${CHECKPOINT_STEM}_${RUN_ID}"
    debug_root="$run_root/debugger"

    inference_cmd=(uv run python -u main_test_vrt.py --opt "$runtime_config")

    debugger_cmd=(
        uv run python -u scripts/analysis/fusion_attribution.py
        --opt "$runtime_config"
        --checkpoint "$checkpoint"
        --samples "$SAMPLES_FILE"
        --out "$debug_root"
        --device cuda:0
        --cam-method "$CAM_METHOD"
        --analysis-num-frames "$ANALYSIS_NUM_FRAMES"
        --analysis-crop-size "$ANALYSIS_CROP_SIZE"
        --analysis-tile-stride "$ANALYSIS_TILE_STRIDE"
        --cam-scopes fullframe roi
    )
    if [[ -n "$MAX_SAMPLES" ]]; then
        debugger_cmd+=(--max-samples "$MAX_SAMPLES")
    fi

    echo ""
    echo "=========================================="
    echo "Forward debug run"
    echo "Experiment: $exp_dir"
    echo "Base config: $base_config"
    echo "Runtime config: $runtime_config"
    echo "Checkpoint: $checkpoint"
    echo "Samples: $SAMPLES_FILE"
    echo "Assigned GPU: $assigned_gpu"
    echo "CAM method: $CAM_METHOD"
    echo "Analysis num frames: $ANALYSIS_NUM_FRAMES"
    echo "Analysis crop size: $ANALYSIS_CROP_SIZE"
    echo "Analysis tile stride: $ANALYSIS_TILE_STRIDE"
    echo "Max samples: ${MAX_SAMPLES:-<all samples>}"
    echo "Stage: $STAGE"
    echo "Full inference output: $inference_result_root"
    echo "Debugger output: $debug_root"
    echo "=========================================="

    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$STAGE" == "both" || "$STAGE" == "inference" ]]; then
            echo "DRY RUN: CUDA_VISIBLE_DEVICES=$assigned_gpu ${inference_cmd[*]}"
        fi
        if [[ "$STAGE" == "both" || "$STAGE" == "debugger" ]]; then
            echo "DRY RUN: CUDA_VISIBLE_DEVICES=$assigned_gpu ${debugger_cmd[*]}"
        fi
        return 0
    fi

    if [[ "$STAGE" == "both" || "$STAGE" == "inference" ]]; then
        echo "[STAGE] full inference: $exp_dir"
        set +e
        CUDA_VISIBLE_DEVICES="$assigned_gpu" "${inference_cmd[@]}"
        cmd_exit=$?
        set -e
        if [[ "$cmd_exit" -ne 0 ]]; then
            return "$cmd_exit"
        fi
    fi

    if [[ "$STAGE" == "both" || "$STAGE" == "debugger" ]]; then
        echo "[STAGE] fusion debugger: $exp_dir"
        set +e
        CUDA_VISIBLE_DEVICES="$assigned_gpu" "${debugger_cmd[@]}"
        cmd_exit=$?
        set -e
        if [[ "$cmd_exit" -ne 0 ]]; then
            return "$cmd_exit"
        fi
    fi

    return 0
}

run_sequential_experiments() {
    local failures_ref="$1"
    local exp_dir assigned_gpu
    local idx=0
    for exp_dir in "${EXPERIMENTS[@]}"; do
        assigned_gpu="${GPU_ID_ARRAY[$((idx % GPU_COUNT))]}"
        if run_one_experiment "$exp_dir" "$assigned_gpu"; then
            echo "[OK] $exp_dir"
        else
            echo "[FAILED] $exp_dir" >&2
            eval "$failures_ref+=(\"\$exp_dir\")"
            if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                return 1
            fi
        fi
        idx=$((idx + 1))
    done
}

run_parallel_experiments() {
    local failures_ref="$1"
    local tmp_dir exp_dir assigned_gpu slot pid status finished_pid log_file
    local next_idx=0
    local running=0
    tmp_dir="$(mktemp -d /tmp/svrt_forward_debug.XXXXXX)"

    declare -a slot_pids=()
    declare -a slot_exps=()
    declare -a slot_logs=()

    while [[ "$next_idx" -lt "${#EXPERIMENTS[@]}" || "$running" -gt 0 ]]; do
        for ((slot = 0; slot < TOTAL_WORKER_SLOTS && next_idx < ${#EXPERIMENTS[@]}; slot++)); do
            if [[ -n "${slot_pids[$slot]:-}" ]]; then
                continue
            fi
            exp_dir="${EXPERIMENTS[$next_idx]}"
            assigned_gpu="${GPU_ID_ARRAY[$((slot % GPU_COUNT))]}"
            log_file="$tmp_dir/exp_${next_idx}.log"
            (
                run_one_experiment "$exp_dir" "$assigned_gpu"
            ) >"$log_file" 2>&1 &
            pid=$!
            slot_pids[$slot]="$pid"
            slot_exps[$slot]="$exp_dir"
            slot_logs[$slot]="$log_file"
            running=$((running + 1))
            echo "[LAUNCHED] gpu=$assigned_gpu pid=$pid exp=$exp_dir log=$log_file"
            next_idx=$((next_idx + 1))
        done

        if [[ "$running" -eq 0 ]]; then
            break
        fi

        set +e
        wait -n -p finished_pid
        status=$?
        set -e
        if [[ "$status" -eq 127 ]]; then
            break
        fi

        for ((slot = 0; slot < TOTAL_WORKER_SLOTS; slot++)); do
            pid="${slot_pids[$slot]:-}"
            if [[ -z "$pid" || "$pid" != "$finished_pid" ]]; then
                continue
            fi
            exp_dir="${slot_exps[$slot]}"
            log_file="${slot_logs[$slot]}"
            echo ""
            echo "----- worker log: $exp_dir -----"
            cat "$log_file"
            echo "----- end worker log: $exp_dir -----"
            if [[ "$status" -eq 0 ]]; then
                echo "[OK] $exp_dir"
            else
                echo "[FAILED] $exp_dir" >&2
                eval "$failures_ref+=(\"\$exp_dir\")"
                if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                    for live_pid in "${slot_pids[@]}"; do
                        if [[ -n "$live_pid" && "$live_pid" != "$finished_pid" ]]; then
                            kill "$live_pid" >/dev/null 2>&1 || true
                        fi
                    done
                    return 1
                fi
            fi
            slot_pids[$slot]=""
            slot_exps[$slot]=""
            slot_logs[$slot]=""
            running=$((running - 1))
            break
        done
    done
}

echo "=========================================="
echo "S-VRT Batch Forward Debug"
echo "=========================================="
echo "Run ID: $RUN_ID"
echo "GPU count: $GPU_COUNT"
echo "GPU list: $GPU_LIST"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total worker slots: $TOTAL_WORKER_SLOTS"
echo "Checkpoint stem: $CHECKPOINT_STEM"
echo "Samples file: $SAMPLES_FILE"
echo "Analysis dataset source: $ANALYSIS_DATASET_SOURCE"
echo "Test dataloader workers: $TEST_NUM_WORKERS"
echo "Max samples: ${MAX_SAMPLES:-<all samples>}"
echo "CAM method: $CAM_METHOD"
echo "Analysis num frames: $ANALYSIS_NUM_FRAMES"
echo "Analysis crop size: $ANALYSIS_CROP_SIZE"
echo "Analysis tile stride: $ANALYSIS_TILE_STRIDE"
echo "Artifact root: ${ARTIFACT_ROOT:-<experiment images/forward_debug>}"
echo "Stage: $STAGE"
echo "Parallel mode: $PARALLEL_MODE"
echo "Continue on error: $CONTINUE_ON_ERROR"
echo "Dry run: $DRY_RUN"
echo "Experiments:"
printf '  %s\n' "${EXPERIMENTS[@]}"
echo ""

failures=()
if [[ "$PARALLEL_MODE" == true && "$GPU_COUNT" -gt 1 ]]; then
    run_parallel_experiments failures || true
else
    run_sequential_experiments failures || true
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
    echo ""
    echo "Failed experiments:"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi

echo ""
echo "All forward debug runs completed."
