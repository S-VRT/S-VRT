from models.select_network import apply_compile_checkpoint_policy


def test_compile_checkpoint_policy_disabled_when_compile_off():
    opt = {
        "train": {
            "compile": {
                "enable": False,
                "checkpoint_compat": {
                    "disable_ffn_blocks": [0],
                    "disable_attn_blocks": [6],
                },
            }
        },
        "netG": {
            "no_checkpoint_ffn_blocks": [1, 2],
            "no_checkpoint_attn_blocks": [3],
        },
    }

    summary = apply_compile_checkpoint_policy(opt)

    assert summary["applied"] is False
    assert opt["netG"]["no_checkpoint_ffn_blocks"] == [1, 2]
    assert opt["netG"]["no_checkpoint_attn_blocks"] == [3]


def test_compile_checkpoint_policy_skips_fusion_only_scope():
    opt = {
        "train": {
            "compile": {
                "enable": True,
                "scope": "fusion_only",
                "checkpoint_compat": {
                    "enable": True,
                    "disable_ffn_blocks": [0],
                    "disable_attn_blocks": [1],
                },
            }
        },
        "netG": {
            "no_checkpoint_ffn_blocks": [2],
            "no_checkpoint_attn_blocks": [3],
        },
    }

    summary = apply_compile_checkpoint_policy(opt)

    assert summary["applied"] is False
    assert opt["netG"]["no_checkpoint_ffn_blocks"] == [2]
    assert opt["netG"]["no_checkpoint_attn_blocks"] == [3]


def test_compile_checkpoint_policy_merges_unique_blocks_when_enabled():
    opt = {
        "rank": 0,
        "train": {
            "compile": {
                "enable": True,
                "checkpoint_compat": {
                    "enable": True,
                    "disable_ffn_blocks": [0, 2, 2],
                    "disable_attn_blocks": [0, 3],
                },
            }
        },
        "netG": {
            "no_checkpoint_ffn_blocks": [1, 2],
            "no_checkpoint_attn_blocks": [3],
        },
    }

    summary = apply_compile_checkpoint_policy(opt)

    assert summary == {
        "applied": True,
        "disable_ffn_blocks": [0, 2],
        "disable_attn_blocks": [0, 3],
        "no_checkpoint_ffn_blocks": [0, 1, 2],
        "no_checkpoint_attn_blocks": [0, 3],
    }
    assert opt["netG"]["no_checkpoint_ffn_blocks"] == [0, 1, 2]
    assert opt["netG"]["no_checkpoint_attn_blocks"] == [0, 3]
