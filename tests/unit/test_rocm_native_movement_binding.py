"""Host-free binding tests: native calls are mocked; no device execution claim.
Optional native movement binding has an explicit context teardown boundary."""
from types import SimpleNamespace
import pytest
from tessera import runtime as rt


def test_disabled_movement_does_not_acquire_cached_library(monkeypatch):
    retained=object()
    monkeypatch.setattr(rt,"_rocm_native_movement_runtime",retained)
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_MOVEMENT","0")
    assert rt._load_rocm_native_movement_runtime() is None
    assert rt._rocm_native_movement_runtime is retained


def test_explicit_missing_library_cannot_select_another_build(monkeypatch,tmp_path):
    monkeypatch.setattr(rt,"_rocm_native_movement_runtime",None)
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_MOVEMENT","1")
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_MOVEMENT_LIB",str(tmp_path/"missing.so"))
    with pytest.raises(RuntimeError,match="configured.*missing"):
        rt._load_rocm_native_movement_runtime()


def test_context_clear_frees_staging_before_unloading_images_when_disabled(monkeypatch):
    calls=[]
    def clear_staging():
        calls.append("staging")
        return 0
    def clear_images():
        calls.append("images")
        return 0
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_MOVEMENT","0")
    monkeypatch.setenv("TESSERA_ROCM_NATIVE_IMAGE_CACHE","0")
    monkeypatch.setattr(rt,"_rocm_native_movement_runtime",
                        SimpleNamespace(tessera_rocm_movement_clear_current=clear_staging))
    monkeypatch.setattr(rt,"_rocm_native_image_runtime",
                        SimpleNamespace(tessera_rocm_image_clear_current=clear_images))
    rt._clear_rocm_native_image_cache()
    assert calls==["staging","images"]


@pytest.mark.parametrize("status",[7,8,9])
def test_failed_staging_clear_preserves_image_leases(monkeypatch,status):
    calls=[]
    def images():
        calls.append("images")
        return 0
    monkeypatch.setattr(rt,"_rocm_native_movement_runtime",
                        SimpleNamespace(tessera_rocm_movement_clear_current=lambda:status))
    monkeypatch.setattr(rt,"_rocm_native_image_runtime",
                        SimpleNamespace(tessera_rocm_image_clear_current=images))
    with pytest.raises(RuntimeError,match="movement clear failed"):
        rt._clear_rocm_native_image_cache()
    assert calls==[]
