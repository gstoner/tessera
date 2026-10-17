"""Exact native provenance names and immutable witness digest reuse."""
from dataclasses import replace
import hashlib

import pytest
from tessera.compiler import native_attention_rows as rows
from tests.unit.test_native_attention_rows_program import package  # noqa: F401


@pytest.mark.parametrize("name",["tesseraXattention_ad_pair","nottessera.attention_ad_pair","other.tessera.attention_ad_pair"])
def test_lookalike_witness_names_do_not_reuse_canonical_source_identity(name):
    rows.pair_witness('tessera.attention_ad_pair = "source"')
    with pytest.raises(ValueError,match="witness"):
        rows.pair_witness(name+' = "source"')


@pytest.mark.compiler_route
def test_repinned_member_with_noncanonical_lineage_name_is_rejected(package):
    program=package()
    component=program.correction
    tile=component.tile_ir.replace('tessera.attention_ad_pair','tesseraXattention_ad_pair')
    target=component.target_ir.replace('tessera.attention_ad_pair','tesseraXattention_ad_pair')
    image=replace(component.image,target_ir_digest=hashlib.sha256(target.encode()).hexdigest())
    descriptor=replace(component.descriptor,image_digest=image.image_digest,
        provenance={**component.descriptor.provenance,
            "tile_ir_digest":hashlib.sha256(tile.encode()).hexdigest(),
            "target_ir_digest":hashlib.sha256(target.encode()).hexdigest()})
    altered=replace(component,tile_ir=tile,target_ir=target,image=image,descriptor=descriptor)
    with pytest.raises(ValueError,match="witness"):
        replace(program,correction=altered).validate()


def witness(encoded):
    return 'tessera.attention_ad_pair = "'+encoded+'"'


@pytest.mark.parametrize("encoded,raw",[
    ("",b""),("plain",b"plain"),("éβ🙂","éβ🙂".encode()),
    (r'quote\"slash\\',b'quote"slash\\'),(r'\00\0a\22\5C',b'\x00\n"\\'),
    (r'\ff\80',b'\xff\x80'),
])
def test_witness_unescape_preserves_exact_bytes(encoded,raw):
    assert rows.pair_witness(witness(encoded))==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("encoded",[r'\q',r'\z0',r'\0','\\'])
def test_invalid_encodings_never_seed_a_digest(encoded):
    rows._WITNESS_DIGEST_CACHE.cache_clear()
    for _ in range(2):
        with pytest.raises(ValueError):rows.pair_witness(witness(encoded))
    assert rows._WITNESS_DIGEST_CACHE.cache_info().currsize==0


def test_equal_encoded_source_reuses_digest_across_different_members():
    rows._WITNESS_DIGEST_CACHE.cache_clear()
    a=rows.pair_witness('module attributes {'+witness('source')+'} {}')
    b=rows.pair_witness('other attributes {'+witness('source')+'} {}')
    assert a==b
    info=rows._WITNESS_DIGEST_CACHE.cache_info()
    assert info.misses==1 and info.hits==1
    assert rows.pair_witness(witness('changed'))!=a


def test_count_and_utf8_content_limits_fall_back_without_skipping_validation(monkeypatch):
    rows._WITNESS_DIGEST_CACHE.cache_clear()
    monkeypatch.setattr(rows,"_WITNESS_CACHE_BYTE_LIMIT",7)
    for encoded in ("abcdefgh","éééé"):
        for _ in range(2):assert rows.pair_witness(witness(encoded))==hashlib.sha256(encoded.encode()).hexdigest()
    assert rows._WITNESS_DIGEST_CACHE.cache_info().currsize==0
    monkeypatch.setattr(rows,"_WITNESS_CACHE_BYTE_LIMIT",256*1024)
    for i in range(40):rows.pair_witness(witness(str(i)))
    assert rows._WITNESS_DIGEST_CACHE.cache_info().currsize==32


def test_process_change_replaces_cache_before_lookup(monkeypatch):
    rows.pair_witness(witness('source'))
    prior=rows._WITNESS_DIGEST_CACHE
    pid=rows.os.getpid()
    monkeypatch.setattr(rows.os,"getpid",lambda:pid+1)
    assert rows.pair_witness(witness('source'))==hashlib.sha256(b'source').hexdigest()
    assert rows._WITNESS_DIGEST_CACHE is not prior
    assert rows._WITNESS_DIGEST_CACHE.cache_info().misses==1


def test_concurrent_calls_preserve_content_and_entry_bound():
    from concurrent.futures import ThreadPoolExecutor
    values=[str(i%10) for i in range(100)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual=list(pool.map(lambda value:rows.pair_witness(witness(value)),values))
    assert actual==[hashlib.sha256(value.encode()).hexdigest() for value in values]
    assert rows._WITNESS_DIGEST_CACHE.cache_info().currsize<=32


def test_duplicate_or_missing_witness_is_rejected_on_hot_cache():
    rows.pair_witness(witness('source'))
    for text in ('module {}',witness('source')+' '+witness('source')):
        with pytest.raises(ValueError,match='one retained'):rows.pair_witness(text)
