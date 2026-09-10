from tessera.compiler.native_driver_isolation import DriverIsolationLease


class Process:
    def __init__(self): self.code=None; self.calls=[]
    def poll(self): return self.code
    def terminate(self): self.calls.append('terminate'); self.code=-15
    def wait(self, timeout=None): self.calls.append('wait'); return self.code


def test_uncertain_driver_context_is_reclaimed_by_process_death_once():
    process=Process(); lease=DriverIsolationLease(process,context_identity='worker-1')
    lease.mark_uncertain(); lease.recover(); lease.recover()
    assert process.calls == ['terminate','wait'] and lease.reusable


def test_healthy_isolation_cannot_be_destroyed_as_recovery():
    lease=DriverIsolationLease(Process(),context_identity='worker-1')
    try: lease.recover()
    except ValueError as error: assert 'uncertain' in str(error)
    else: raise AssertionError('healthy isolation destroyed')


def test_stalled_isolation_is_killed_after_bounded_terminate():
    import subprocess
    class Stalled(Process):
        def wait(self, timeout=None):
            self.calls.append(('wait',timeout))
            if len(self.calls)==2: raise subprocess.TimeoutExpired('worker',timeout)
            return self.code
        def terminate(self): self.calls.append('terminate')
        def kill(self): self.calls.append('kill'); self.code=-9
    process=Stalled(); lease=DriverIsolationLease(process,context_identity='worker',timeout_seconds=.25)
    lease.mark_uncertain(); lease.recover()
    assert process.calls == ['terminate',('wait',.25),'kill',('wait',.25)]
