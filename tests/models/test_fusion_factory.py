import pytest

from models.fusion.factory import create_fusion_operator


def test_create_fusion_operator_concat():
    op = create_fusion_operator('concat', placement='replace')
    assert op is not None


def test_create_fusion_operator_unknown():
    with pytest.raises(ValueError, match='Unknown fusion operator'):
        create_fusion_operator('missing')


def test_create_fusion_operator_unknown_placement():
    with pytest.raises(ValueError, match='Unknown fusion placement'):
        create_fusion_operator('concat', placement='missing')
