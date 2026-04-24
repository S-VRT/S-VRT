# Simplified Input Strategy Config Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign S-VRT configuration so `concat` and `fusion` become first-class alternative input strategies, while preserving legacy fields through explicit compatibility readers.

**Architecture:** Introduce canonical config fields under `netG.input` to separate model input strategy (`concat | fusion`), input packing mode (`concat | dual`), and raw ingress width. Keep `netG.concat` and `netG.fusion` as peer namespaces for strategy-specific options, and reserve `compat` only for actual migration aids such as `keep_legacy_L` and legacy field fallback.

**Tech Stack:** Python, PyTorch config loading paths, pytest, JSON option files

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `data/dataset_video_train_rgbspike.py` | Modify | Read canonical spike bins and `compat.keep_legacy_L`; keep legacy dataset field fallbacks |
| `data/dataset_video_test.py` | Modify | Mirror canonical dataset config resolution for test/val |
| `models/model_plain.py` | Modify | Resolve canonical `netG.input.{strategy,mode,raw_ingress_chans}` and enforce valid strategy/mode combinations |
| `models/architectures/vrt/vrt.py` | Modify | Resolve canonical strategy/mode/raw ingress width and gate fusion construction off `input.strategy` |
| `models/select_network.py` | Modify | Pass canonical raw ingress width into VRT construction |
| `tests/data/test_dataset_rgbspike_pack_modes.py` | Modify | Add coverage for canonical dataset fields and legacy mismatch behavior |
| `tests/models/test_model_plain_dual_input_feed.py` | Modify | Add canonical `netG.input` coverage and strategy/mode validation tests |
| `tests/models/test_vrt_dual_input_priority.py` | Modify | Add canonical input strategy coverage and invalid combination checks |
| `options/gopro_rgbspike_local.json` | Modify | Adopt canonical `input`, `concat`, `fusion`, and `compat` layout while keeping legacy mirrors |
| `options/gopro_rgbspike_local_debug.json` | Modify | Mirror canonical layout for debug config |
| `options/gopro_rgbspike_server.json` | Modify | Mirror canonical layout for server config |

### Task 1: Canonical Dataset Fields And Compatibility Reader

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`
- Modify: `data/dataset_video_test.py`
- Test: `tests/data/test_dataset_rgbspike_pack_modes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/data/test_dataset_rgbspike_pack_modes.py`:

```python
def test_canonical_num_bins_drives_spike_channels(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        spike_channels=None,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 5}},
    )
    dataset.opt.pop("spike_channels", None)
    assert dataset.spike_channels == 5


def test_compat_keep_legacy_l_overrides_top_level_flag(tmp_path):
    opt = _build_opt(
        tmp_path,
        input_pack_mode="dual",
        keep_legacy_l=True,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 2}},
    )
    opt["compat"] = {"keep_legacy_L": False}
    dataset = TrainDatasetRGBSpike(opt)
    assert dataset.keep_legacy_l is False


def test_legacy_spike_channels_conflicts_with_canonical_num_bins(tmp_path):
    opt = _build_opt(
        tmp_path,
        spike_channels=4,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 3}},
    )
    with pytest.raises(ValueError, match="spike_channels=4"):
        TrainDatasetRGBSpike(opt)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\\.venv\\Scripts\\pytest.exe tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: FAIL because only legacy top-level `keep_legacy_l` is read and canonical `compat.keep_legacy_L` is ignored.

- [ ] **Step 3: Implement canonical dataset config resolution**

In both dataset files, resolve canonical fields with this pattern:

```python
spike_cfg = opt.get("spike", {}) if isinstance(opt.get("spike", {}), dict) else {}
recon_cfg = spike_cfg.get("reconstruction", {})
compat_cfg = opt.get("compat", {}) if isinstance(opt.get("compat", {}), dict) else {}

nested_num_bins = recon_cfg.get("num_bins", None)
self.spike_channels = int(opt.get("spike_channels", nested_num_bins if nested_num_bins is not None else 4))
if nested_num_bins is not None and "spike_channels" in opt and int(opt["spike_channels"]) != int(nested_num_bins):
    raise ValueError(
        f"[TrainDatasetRGBSpike] Conflicting channel settings: spike_channels={int(opt['spike_channels'])} "
        f"vs spike.reconstruction.num_bins={int(nested_num_bins)}."
    )

self.keep_legacy_l = bool(compat_cfg.get("keep_legacy_L", opt.get("keep_legacy_l", True)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\\.venv\\Scripts\\pytest.exe tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: PASS for both canonical and legacy dataset config styles.

- [ ] **Step 5: Commit**

```bash
git add data/dataset_video_train_rgbspike.py data/dataset_video_test.py tests/data/test_dataset_rgbspike_pack_modes.py
git commit -m "feat(config): add canonical dataset config readers"
```

### Task 2: Canonical Input Strategy Reader In ModelPlain

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_model_plain_dual_input_feed.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/models/test_model_plain_dual_input_feed.py`:

```python
def test_build_model_input_uses_canonical_input_mode_dual():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"netG": {"input": {"mode": "dual"}, "in_chans": 7}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    out = model._build_model_input_tensor(
        {"L_rgb": torch.randn(1, 2, 3, 8, 8), "L_spike": torch.randn(1, 2, 4, 8, 8)}
    )
    assert out.shape == (1, 2, 7, 8, 8)


def test_concat_strategy_accepts_concat_mode():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"netG": {"input": {"strategy": "concat", "mode": "concat", "raw_ingress_chans": 7}}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    out = model._build_model_input_tensor({"L": torch.randn(1, 2, 7, 8, 8)})
    assert out.shape == (1, 2, 7, 8, 8)


def test_fusion_strategy_rejects_concat_mode():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"netG": {"input": {"strategy": "fusion", "mode": "concat", "raw_ingress_chans": 11}}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    with pytest.raises(ValueError, match="strategy=fusion"):
        model._resolve_input_mode()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_model_plain_dual_input_feed.py -v`
Expected: FAIL because canonical `netG.input` fields are not fully recognized and strategy/mode coupling is not enforced.

- [ ] **Step 3: Implement canonical input strategy resolution**

In `models/model_plain.py`, introduce helpers that resolve canonical fields first:

```python
def _resolve_input_cfg(self):
    net_cfg = self.opt.get("netG", {})
    input_cfg = net_cfg.get("input", {}) if isinstance(net_cfg.get("input", {}), dict) else {}
    strategy = str(input_cfg.get("strategy", "fusion" if net_cfg.get("fusion", {}).get("enable", False) else "concat")).strip().lower()
    mode = str(input_cfg.get("mode", net_cfg.get("input_mode", "concat"))).strip().lower()
    raw_ingress_chans = int(input_cfg.get("raw_ingress_chans", net_cfg.get("in_chans", 3)))
    return strategy, mode, raw_ingress_chans
```

Then enforce:

```python
if strategy == "fusion" and mode != "dual":
    raise ValueError("input.strategy=fusion requires input.mode='dual'.")
if strategy == "concat" and mode not in {"concat", "dual"}:
    raise ValueError("input.strategy=concat supports input.mode 'concat' or 'dual'.")
```

Use `raw_ingress_chans` inside `_assert_lq_channels`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_model_plain_dual_input_feed.py -v`
Expected: PASS with both canonical and legacy field styles.

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py tests/models/test_model_plain_dual_input_feed.py
git commit -m "feat(config): add canonical input strategy reader to model plain"
```

### Task 3: Canonical Input Strategy Reader In VRT

**Files:**
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `models/select_network.py`
- Test: `tests/models/test_vrt_dual_input_priority.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/models/test_vrt_dual_input_priority.py`:

```python
def test_vrt_uses_canonical_input_strategy_and_mode():
    opt = {
        "netG": {
            "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 11},
            "fusion": {
                "placement": "early",
                "operator": "concat",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    model = _build_vrt(opt=opt, in_chans=11)
    assert model.input_mode == "dual"
    assert model.in_chans == 11


def test_vrt_rejects_fusion_strategy_with_concat_mode():
    opt = {
        "netG": {
            "input": {"strategy": "fusion", "mode": "concat", "raw_ingress_chans": 11},
            "fusion": {
                "placement": "early",
                "operator": "concat",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }
    with pytest.raises(ValueError, match="strategy=fusion"):
        _build_vrt(opt=opt, in_chans=11)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_vrt_dual_input_priority.py -v`
Expected: FAIL because VRT still keys off `fusion.enable` and legacy `input_mode`.

- [ ] **Step 3: Implement canonical input strategy in VRT**

Use this resolution pattern near the start of `VRT.__init__`:

```python
net_cfg = (opt or {}).get("netG", {}) or {}
input_cfg = net_cfg.get("input", {}) if isinstance(net_cfg.get("input", {}), dict) else {}
raw_strategy = input_cfg.get("strategy", "fusion" if (net_cfg.get("fusion", {}) or {}).get("enable", False) else "concat")
self.input_strategy = str(raw_strategy).strip().lower()
raw_input_mode = input_cfg.get("mode", net_cfg.get("input_mode", "concat"))
self.input_mode = str(raw_input_mode).strip().lower()
resolved_in_chans = int(input_cfg.get("raw_ingress_chans", in_chans))
self.in_chans = resolved_in_chans
```

Then enforce:

```python
if self.input_strategy == "fusion" and self.input_mode != "dual":
    raise ValueError("[VRT] input.strategy=fusion requires input.mode='dual'.")
```

And derive fusion enablement like:

```python
self.fusion_enabled = self.input_strategy == "fusion"
```

while still allowing legacy `fusion.enable` to backfill `input.strategy` when canonical fields are absent.

- [ ] **Step 4: Update `models/select_network.py` to pass canonical raw ingress width**

Use:

```python
input_cfg = opt_net.get("input", {}) if isinstance(opt_net.get("input", {}), dict) else {}
raw_ingress_chans = int(input_cfg.get("raw_ingress_chans", opt_net.get("in_chans", 3)))
```

and pass `in_chans=raw_ingress_chans`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_vrt_dual_input_priority.py -v`
Expected: PASS for canonical and legacy config styles.

- [ ] **Step 6: Commit**

```bash
git add models/architectures/vrt/vrt.py models/select_network.py tests/models/test_vrt_dual_input_priority.py
git commit -m "feat(config): add canonical input strategy reader to vrt"
```

### Task 4: Peer Namespaces In Options Files

**Files:**
- Modify: `options/gopro_rgbspike_local.json`
- Modify: `options/gopro_rgbspike_local_debug.json`
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: Add canonical `netG.input` block**

Add:

```json
"input": {
  "strategy": "fusion",
  "mode": "dual",
  "raw_ingress_chans": 11
}
```

- [ ] **Step 2: Keep `concat` and `fusion` as peer namespaces**

Use the following shape:

```json
"concat": {},
"fusion": {
  "placement": "early",
  "operator": "gated",
  "out_chans": 3,
  "operator_params": {}
}
```

For concat baselines, switch to:

```json
"input": {
  "strategy": "concat",
  "mode": "concat",
  "raw_ingress_chans": 11
},
"concat": {}
```

- [ ] **Step 3: Move true compatibility-only fields under `compat`**

Use:

```json
"compat": {
  "keep_legacy_L": true
}
```

Keep legacy mirrors for now:

```json
"keep_legacy_l": true,
"input_mode": "dual",
"in_chans": 11,
"spike_channels": 8,
"spike_reconstruction": {"type": "spikecv_tfp"}
```

- [ ] **Step 4: Validate JSON parses cleanly**

Run: `python - <<'PY'\nimport json5, pathlib\nfor path in [pathlib.Path('options/gopro_rgbspike_local.json'), pathlib.Path('options/gopro_rgbspike_local_debug.json'), pathlib.Path('options/gopro_rgbspike_server.json')]:\n    json5.load(open(path, 'r', encoding='utf-8'))\n    print(path.name, 'OK')\nPY`
Expected: each file prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add options/gopro_rgbspike_local.json options/gopro_rgbspike_local_debug.json options/gopro_rgbspike_server.json
git commit -m "config: add canonical input strategy layout"
```

### Task 5: Messaging And Naming Cleanup

**Files:**
- Modify: `models/model_plain.py`
- Modify: `data/dataset_video_train_rgbspike.py`
- Modify: `data/dataset_video_test.py`

- [ ] **Step 1: Write the failing assertions**

Add targeted assertions in existing tests for wording such as:

```python
with pytest.raises(ValueError, match="raw ingress contract"):
    model._assert_lq_channels(torch.randn(1, 2, 7, 8, 8), "Training Feed Data")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_model_plain_dual_input_feed.py tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: FAIL because runtime messaging still describes legacy semantics too loosely.

- [ ] **Step 3: Update wording to match the new model**

Use wording like:

```python
"""Validate the raw model-ingress tensor against configured input width."""
...
f"Hint: raw ingress width validates the tensor before concat/fusion strategy-specific handling "
f"(e.g. RGB 3 + spike bins 8 = 11)."
```

Dataset docstrings should describe:
- `spike_channels` as a legacy alias for reconstruction bins
- `concat` as a first-class strategy, not a compatibility path

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\\.venv\\Scripts\\pytest.exe tests/models/test_model_plain_dual_input_feed.py tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: PASS with updated wording.

- [ ] **Step 5: Commit**

```bash
git add models/model_plain.py data/dataset_video_train_rgbspike.py data/dataset_video_test.py tests/models/test_model_plain_dual_input_feed.py tests/data/test_dataset_rgbspike_pack_modes.py
git commit -m "docs(config): align messaging with input strategy semantics"
```

## Self-Review

1. **Spec coverage:** This plan covers the simplified redesign requested here: `concat` and `fusion` are peer strategies; `compat` only handles actual migration aids; canonical fields live under `netG.input`, `netG.concat`, `netG.fusion`, `datasets.*.spike`, and `datasets.*.compat`.
2. **Placeholder scan:** No placeholder tasks remain; every task names exact files, tests, and commands.
3. **Type consistency:** Canonical field names are consistent across the plan: `netG.input.strategy`, `netG.input.mode`, `netG.input.raw_ingress_chans`, `netG.concat`, `netG.fusion`, `datasets.*.spike.reconstruction.num_bins`, and `datasets.*.compat.keep_legacy_L`.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-15-simplified-config-semantics-migration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
