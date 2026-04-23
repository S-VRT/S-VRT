from models.model_vrt import (
    apply_validation_checkpointing,
    resolve_lazy_flow_cache_mode,
    restore_validation_checkpointing,
    should_score_validation_batch,
    split_patch_coords_for_rank,
)


def test_split_patch_coords_for_rank_stripes_work_without_duplicates():
    coords = [(0, 0), (0, 4), (4, 0), (4, 4), (8, 0)]

    rank0 = split_patch_coords_for_rank(coords, rank=0, world_size=2)
    rank1 = split_patch_coords_for_rank(coords, rank=1, world_size=2)

    assert rank0 == [(0, 0), (4, 0), (8, 0)]
    assert rank1 == [(0, 4), (4, 4)]
    assert sorted(rank0 + rank1) == sorted(coords)


def test_split_patch_coords_for_rank_keeps_all_coords_when_single_rank():
    coords = [(0, 0), (0, 4)]

    assert split_patch_coords_for_rank(coords, rank=0, world_size=1) == coords


def test_should_score_validation_batch_only_rank0_for_distributed_patch_mode():
    opt = {"dist": True, "rank": 1, "val": {"distributed_patch_testing": True}}

    assert not should_score_validation_batch(opt)
    assert should_score_validation_batch({**opt, "rank": 0})


def test_should_score_validation_batch_keeps_existing_distributed_behavior_by_default():
    opt = {"dist": True, "rank": 1, "val": {}}

    assert should_score_validation_batch(opt)


def test_should_score_validation_batch_uses_runtime_active_flag_when_present():
    opt = {
        "dist": True,
        "rank": 1,
        "val": {
            "distributed_patch_testing": True,
            "distributed_patch_testing_active": False,
        },
    }

    assert should_score_validation_batch(opt)


def test_resolve_lazy_flow_cache_mode_prefers_explicit_mode():
    val_opt = {"cache_flow_patches_cpu": True, "lazy_flow_cache_mode": "gpu_clip"}

    assert resolve_lazy_flow_cache_mode(val_opt) == "gpu_clip"


def test_resolve_lazy_flow_cache_mode_keeps_legacy_cpu_patch_flag():
    assert resolve_lazy_flow_cache_mode({"cache_flow_patches_cpu": True}) == "cpu_patch"
    assert resolve_lazy_flow_cache_mode({"cache_flow_patches_cpu": False}) == "none"


def test_resolve_lazy_flow_cache_mode_rejects_unknown_mode():
    try:
        resolve_lazy_flow_cache_mode({"lazy_flow_cache_mode": "banana"})
    except ValueError as exc:
        assert "lazy_flow_cache_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid lazy_flow_cache_mode")


class _CheckpointLeaf:
    def __init__(self):
        self.use_checkpoint_attn = True
        self.use_checkpoint_ffn = True

    def modules(self):
        return [self]


def test_validation_checkpointing_can_disable_and_restore_attrs():
    model = _CheckpointLeaf()

    state = apply_validation_checkpointing(model, disable=True)

    assert model.use_checkpoint_attn is False
    assert model.use_checkpoint_ffn is False
    restore_validation_checkpointing(state)
    assert model.use_checkpoint_attn is True
    assert model.use_checkpoint_ffn is True
