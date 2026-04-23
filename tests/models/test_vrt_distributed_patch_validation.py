from models.model_vrt import (
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
