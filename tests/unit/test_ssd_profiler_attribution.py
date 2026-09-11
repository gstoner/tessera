import sqlite3
import pytest
from benchmarks.attribute_ssd_cuda import attribute


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / 'capture.sqlite'
    with sqlite3.connect(path) as c:
        c.executescript('''
        CREATE TABLE TARGET_INFO_GPU(id,computeMajor,computeMinor);
        INSERT INTO TARGET_INFO_GPU VALUES(0,12,0);
        CREATE TABLE TARGET_INFO_CUDA_DEVICE(cudaId,gpuId,uuid,pid);
        INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(0,0,'test-device',42);
        CREATE TABLE StringIds(id,value);
        INSERT INTO StringIds VALUES(1,'cuLaunchKernel'),(2,'product');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,correlationId,globalPid,deviceId,demangledName);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,correlationId,globalTid,returnValue,nameId);
        ''')
        for i in range(701):
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,0,2)',
                      (i*100+20,i*100+30,i,42<<24))
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,0,1)',
                      (i*100,i*100+10,i,(42<<24)+9))
    packet = dict(backend='nvidia',execution='native_gpu',process_id=42,architecture='sm_120',run_id='a'*32,
                  rows=[dict(device_event_ms=[1]*7,binding_digest='b'*64,image_sha256='c'*64)])
    return path, packet


def test_correlates_kernels_to_process_and_successful_api(capture):
    path, packet = capture
    result = attribute(packet, path)
    assert result['matched_launch_count'] == result['profiler_kernel_count'] == 701
    assert not result['promotion_eligible']


@pytest.mark.parametrize('mutation', [
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=7 WHERE correlationId=1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=99 WHERE correlationId=1',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=2 WHERE correlationId=1',
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1',
    "UPDATE StringIds SET value='other_kernel' WHERE id=2",
    'UPDATE TARGET_INFO_CUDA_DEVICE SET pid=43',
])
def test_refuses_incomplete_or_misattributed_capture(capture, mutation):
    path, packet = capture
    with sqlite3.connect(path) as c:
        c.execute(mutation)
    with pytest.raises(ValueError):
        attribute(packet, path)
