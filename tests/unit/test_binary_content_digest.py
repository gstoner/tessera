"""Tool reuse must invalidate on rebuilds even when size/mtime are preserved."""
import hashlib
import os
from tessera.compiler.toolchain_identity import binary_content_digest


def test_stable_tool_and_same_size_inplace_rebuild(tmp_path):
    path=tmp_path/'compiler'
    path.write_bytes(b'original')
    before=path.stat()
    assert binary_content_digest(path)==hashlib.sha256(b'original').hexdigest()
    assert binary_content_digest(path)==hashlib.sha256(b'original').hexdigest()
    path.write_bytes(b'replaced')
    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert binary_content_digest(path)==hashlib.sha256(b'replaced').hexdigest()


def test_atomic_replacement_and_symlink_retarget(tmp_path):
    path=tmp_path/'compiler'; other=tmp_path/'replacement'; alias=tmp_path/'alias'
    path.write_bytes(b'original'); other.write_bytes(b'replaced')
    before=path.stat()
    alias.symlink_to(path)
    assert binary_content_digest(alias)==hashlib.sha256(b'original').hexdigest()
    os.utime(other,ns=(before.st_atime_ns,before.st_mtime_ns))
    os.replace(other,path)
    assert binary_content_digest(alias)==hashlib.sha256(b'replaced').hexdigest()
    other.write_bytes(b'targeted')
    alias.unlink();alias.symlink_to(other)
    assert binary_content_digest(alias)==hashlib.sha256(b'targeted').hexdigest()


def test_mutation_during_digest_is_not_cached(tmp_path,monkeypatch):
    from tessera.compiler import toolchain_identity as identity
    path=tmp_path/'compiler';path.write_bytes(b'original')
    sha=identity.hashlib.sha256
    class MutatingHash:
        def __init__(self): self.hash=sha()
        def update(self,data):
            self.hash.update(data)
            path.write_bytes(b'replaced')
        def hexdigest(self): return self.hash.hexdigest()
    monkeypatch.setattr(identity.hashlib,'sha256',MutatingHash)
    import pytest
    with pytest.raises(RuntimeError,match='changed while computing'):
        binary_content_digest(path)
    monkeypatch.setattr(identity.hashlib,'sha256',sha)
    assert binary_content_digest(path)==hashlib.sha256(b'replaced').hexdigest()
