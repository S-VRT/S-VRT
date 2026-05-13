from types import SimpleNamespace

from main_train_vrt import maybe_dump_full_frame_fusion_debug_from_batch


class _DummyDumper:
    def __init__(self, should_dump):
        self._should_dump = should_dump

    def should_dump_phase1_last(self, current_step, fix_iter, source=None):
        return self._should_dump and current_step == fix_iter - 1 and source == "val_full_frame"


def test_val_full_frame_debug_uses_first_validation_batch_only():
    calls = []

    class _DummyModel:
        fix_iter = 10
        fusion_debug = _DummyDumper(True)

        def dump_full_frame_fusion_only_from_batch(self, batch, current_step):
            calls.append((batch, current_step))
            return True

    model = _DummyModel()
    batch = {"folder": "clip0"}

    assert maybe_dump_full_frame_fusion_debug_from_batch(model, batch, current_step=9, batch_idx=0) is True
    assert calls == [(batch, 9)]


def test_val_full_frame_debug_skips_non_first_validation_batches():
    calls = []

    class _DummyModel:
        fix_iter = 10
        fusion_debug = _DummyDumper(True)

        def dump_full_frame_fusion_only_from_batch(self, batch, current_step):
            calls.append((batch, current_step))
            return True

    model = _DummyModel()

    assert maybe_dump_full_frame_fusion_debug_from_batch(model, {"folder": "clip1"}, current_step=9, batch_idx=1) is False
    assert calls == []


def test_val_full_frame_debug_skips_when_phase1_trigger_not_active():
    model = SimpleNamespace(
        fix_iter=10,
        fusion_debug=_DummyDumper(False),
        dump_full_frame_fusion_only_from_batch=lambda batch, current_step: True,
    )

    assert maybe_dump_full_frame_fusion_debug_from_batch(model, {"folder": "clip0"}, current_step=9, batch_idx=0) is False
