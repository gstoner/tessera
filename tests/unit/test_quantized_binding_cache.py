"""Complete-content, bounded certificate reuse without stale contract admission."""
from dataclasses import dataclass,replace
import pytest
from tessera.compiler import quantized_binding_cache as cache


@pytest.fixture
def fixture():
    calls=[]
    @dataclass(frozen=True)
    class Package:
        graph: str='verified'
        payload: bytes=b'native image'
        count: int=1
        policy: object=None
        @cache.memoized_validation
        def validate(self):
            calls.append('validate')
            if self.graph!='verified' or self.payload!=b'native image' or type(self.count) is not int:
                raise ValueError('contract differs')
            if self.policy is not None and self.policy['nested']['layout']!='nk':
                raise ValueError('layout differs')
    class Binder:
        @cache.memoized_binding
        def bind(self,package):
            package.validate();calls.append('bind')
            return (package.payload,package.count)
    return Package,Binder,calls


def test_equal_complete_content_reuses_validation_and_binding(fixture):
    Package,Binder,calls=fixture;p=Package(policy={'nested':{'layout':'nk'}})
    binder=Binder();a=binder.bind(p);b=binder.bind(replace(p))
    assert a is b and calls==['validate','bind']
    assert binder.bind.cache_info()['hits']==1


@pytest.mark.parametrize('field,value',[('graph','different'),('payload',b'different'),('count',True)])
def test_changed_frozen_fields_do_not_hit_valid_certificate(fixture,field,value):
    Package,Binder,_=fixture;p=Package();binder=Binder();binder.bind(p)
    with pytest.raises(ValueError):binder.bind(replace(p,**{field:value}))


def test_nested_in_place_metadata_changes_invalidate_both_caches(fixture):
    Package,Binder,_=fixture;p=Package(policy={'nested':{'layout':'nk'}});binder=Binder();binder.bind(p)
    p.policy['nested']['layout']='wrong'
    with pytest.raises(ValueError,match='layout'):p.validate()
    with pytest.raises(ValueError,match='layout'):binder.bind(p)


def test_cache_switch_preserves_uncached_validation_and_binding(fixture,monkeypatch):
    Package,Binder,calls=fixture;monkeypatch.setenv('TESSERA_ROCM_PROGRAM_BINDING_CACHE','0')
    binder=Binder();p=Package();binder.bind(p);binder.bind(p)
    assert calls==['validate','bind','validate','bind']
    assert binder.bind.cache_info()['size']==0


def test_entry_count_is_bounded_and_oldest_is_evicted(fixture):
    Package,Binder,calls=fixture;binder=Binder()
    for i in range(20):binder.bind(Package(count=i))
    assert binder.bind.cache_info()['size']==16
    count=len(calls);binder.bind(Package(count=0))
    assert len(calls)==count+2


def test_oversize_content_uses_full_uncached_path(fixture,monkeypatch):
    Package,Binder,calls=fixture;monkeypatch.setattr(cache,'_ENTRY_BYTES',32)
    p=Package();binder=Binder();binder.bind(p);binder.bind(p)
    assert calls==['validate','bind','validate','bind']
    assert binder.bind.cache_info()['size']==0


def test_mutation_during_validation_cannot_seed_a_certificate():
    @dataclass(frozen=True)
    class Package:
        policy: object
        @cache.memoized_validation
        def validate(self):self.policy['revision']+=1
    p=Package({'revision':0})
    with pytest.raises(ValueError,match='changed during'):p.validate()
    assert p.validate.cache_info()['size']==0


def test_process_change_discards_cached_host_bindings(fixture,monkeypatch):
    Package,Binder,calls=fixture;p=Package();binder=Binder();binder.bind(p)
    pid=cache.os.getpid();monkeypatch.setattr(cache.os,'getpid',lambda:pid+1)
    binder.bind(p)
    assert calls==['validate','bind','validate','bind']
    assert binder.bind.cache_info()['size']==1


def test_keyword_self_cannot_share_a_certificate_with_other_content(fixture):
    Package,_,calls=fixture;p=Package()
    Package.validate(self=p)
    with pytest.raises(ValueError):Package.validate(self=replace(p,graph='wrong'))
    assert calls==['validate','validate']


def test_process_change_resets_inspection_before_lock_access(fixture,monkeypatch):
    Package,Binder,_=fixture;binder=Binder();binder.bind(Package())
    pid=cache.os.getpid();monkeypatch.setattr(cache.os,'getpid',lambda:pid+1)
    assert binder.bind.cache_info()['size']==0
    binder.bind.cache_clear()
    assert binder.bind.cache_info()['size']==0


def test_unicode_content_is_charged_by_utf8_bytes(monkeypatch):
    monkeypatch.setattr(cache,'_ENTRY_BYTES',128)
    with pytest.raises(cache._Uncacheable):cache._snapshot('é'*65)


def test_large_metadata_keys_cannot_escape_the_content_budget(fixture,monkeypatch):
    Package,_,calls=fixture;monkeypatch.setattr(cache,'_ENTRY_BYTES',2048)
    p=Package(policy={'nested':{'layout':'nk'},'é'*1024:None})
    p.validate();p.validate()
    assert calls==['validate','validate']
    assert p.validate.cache_info()['size']==0


def test_signed_zero_float_bits_are_distinct_content(fixture):
    Package,_,calls=fixture;p=Package(policy={'nested':{'layout':'nk'},'zero':0.0})
    p.validate();p.policy['zero']=-0.0;p.validate()
    assert calls==['validate','validate']
