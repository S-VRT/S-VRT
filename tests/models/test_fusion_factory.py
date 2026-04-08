import pytest

from models.fusion.factory import create_fusion_operator, create_fusion_adapter


def test_create_fusion_operator_concat():
    op = create_fusion_operator(
        operator_name='concat',
        rgb_chans=3,
        spike_chans=2,
        out_chans=5,
        operator_params={},
    )
    assert op is not None


def test_create_fusion_operator_unknown():
    with pytest.raises(ValueError, match='Unknown fusion operator'):
        create_fusion_operator(
            operator_name='missing',
            rgb_chans=3,
            spike_chans=2,
            out_chans=5,
            operator_params={},
        )


def test_create_fusion_adapter_unknown_placement():
    with pytest.raises(ValueError, match='Unknown fusion placement'):
        create_fusion_adapter(
            placement='unknown',
            operator=None,
            mode='replace',
            inject_stages=[],
        )
