"""The quantized MLP timing recorders take their architecture from the live device.

GFX1151-QUANTIZED-MLP-TIMING-20261101. Host-free: the live architecture is
monkeypatched, and no recorder is run.
"""
import fcntl
import pathlib

import pytest
from benchmarks.rocm import quantized_mlp_arch as owning
from tessera import runtime as rt

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECORDERS = ("benchmark_public_quantized_mlp.py", "benchmark_public_bounded_quantized_mlp.py",
             "benchmark_k_bounded_quantized_mlp.py", "diagnose_bounded_quantized_mlp.py",
             "attribute_bounded_quantized_mlp_members.py")


@pytest.mark.parametrize("arch", owning.OWNING_ARCHES)
def test_owning_arch_admits_each_owner(monkeypatch, arch):
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: arch)
    assert owning.owning_arch() == arch
    assert owning.target(arch) == "rocm_" + arch
    assert owning.device(arch).startswith(arch + " ")


@pytest.mark.parametrize("arch", ["gfx1100", "gfx942", None])
def test_owning_arch_refuses_other_devices(monkeypatch, arch):
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: arch)
    with pytest.raises(SystemExit, match="requires an owning gfx1201/gfx1151 device"):
        owning.owning_arch()
    with pytest.raises(ValueError):
        owning.target(arch)


def test_source_retargets_exactly_one_target_and_arch():
    gfx1201 = owning.source(3, 32, 48, 19, "gfx1201")
    gfx1151 = owning.source(3, 32, 48, 19, "gfx1151")
    assert '"rocm_gfx1201"' in gfx1201 and "gfx1151" not in gfx1201
    assert "gfx1201" not in gfx1151
    assert gfx1151.count('"rocm_gfx1151"') == 1 and gfx1151.count('tessera.arch = "gfx1151"') == 1
    with pytest.raises(ValueError):
        owning.source(3, 32, 48, 19, "gfx1100")


def test_sync_key_naming_another_arch_is_refused():
    assert owning.check_sync_key("GFX1151-BOUNDED-QUANTIZED-K-20261101", "gfx1151")
    with pytest.raises(SystemExit, match="names gfx1201"):
        owning.check_sync_key("GFX1201-BOUNDED-QUANTIZED-K-20261028", "gfx1151")
    with pytest.raises(SystemExit, match="names gfx1151"):
        owning.check_sync_key("GFX1151-X", "gfx1201")


def test_device_lock_must_be_held(tmp_path):
    path = str(tmp_path / "device.lock")
    assert owning.device_lock(None) is None
    with pytest.raises(SystemExit, match="is not held"):
        owning.device_lock(path)
    with open(path, "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert owning.device_lock(path) == {"path": path, "held": True}


@pytest.mark.parametrize("name", RECORDERS)
def test_recorders_name_no_fixed_architecture(name):
    text = (ROOT / "benchmarks" / "rocm" / name).read_text()
    assert "quantized_mlp_arch" in text
    for literal in ('"rocm_gfx1201"', "'rocm_gfx1201'", '!= "gfx1201"', "!='gfx1201'",
                    'device": "gfx1201'):
        assert literal not in text, (name, literal)
