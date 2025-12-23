"""Smoke tests for flow_builder.compute_flows"""
import torch
from mmvrt.models.motion.flow_builder import compute_flows
from mmvrt.models.motion.spynet import SpyNet


def _make_input(b=1, n=5, c=3, h=32, w=32):
    return torch.rand((b, n, c, h, w), dtype=torch.float32)


def test_compute_flows_pa2():
    x = _make_input()
    fb, ff = compute_flows(x, pa_frames=2, spynet=SpyNet())
    assert isinstance(fb, list) and isinstance(ff, list)
    print("pa2 flows OK")


def test_compute_flows_pa4():
    x = _make_input()
    fb, ff = compute_flows(x, pa_frames=4, spynet=SpyNet())
    assert isinstance(fb, list) and isinstance(ff, list)
    print("pa4 flows OK")


def test_compute_flows_pa6():
    x = _make_input()
    fb, ff = compute_flows(x, pa_frames=6, spynet=SpyNet())
    assert isinstance(fb, list) and isinstance(ff, list)
    print("pa6 flows OK")


if __name__ == "__main__":
    test_compute_flows_pa2()
    test_compute_flows_pa4()
    test_compute_flows_pa6()
    print("flow_builder smoke tests passed")


