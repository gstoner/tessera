import sqlite3
import pytest
from benchmarks.attribute_mixed_cuda import attribute


@pytest.fixture
def capture(tmp_path):
    path=tmp_path/'mixed.sqlite'
    rows=[]
    with sqlite3.connect(path) as c:
        c.executescript('''CREATE TABLE NVTX_EVENTS(start,end,globalTid,text);
        CREATE TABLE StringIds(id,value);
        INSERT INTO StringIds VALUES(1,'cuLaunchKernel'),(2,'same_kernel_name');
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,returnValue,nameId);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,deviceId,contextId,streamId,demangledName);''')
        for i in range(2):
            artifact,image=str(i)*64,str(i+2)*64
            label=f'tessera:{"a"*32}:{artifact}:{image}:0'
            rows.append(dict(label=label,artifact=artifact,image=image,workload=str(i)))
            c.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)',(i*100,i*100+90,(42<<24)+7,label))
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,0,1)',(i*100+10,i*100+20,(42<<24)+7,i))
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,0,1,1,2)',(i*100+30,i*100+40,42<<24,i))
    return path,dict(backend='nvidia',nvtx=True,execution_kind='native_gpu',ok=True,process_id=42,run_id='a'*32,rows=rows)


def test_same_kernel_name_maps_to_distinct_artifacts(capture):
    path,packet=capture
    result=attribute(packet,path)
    assert [r['artifact'] for r in result['rows']] == ['0'*64,'1'*64]
    assert all(r['kernel_count']==1 for r in result['rows'])
    assert not result['promotion_eligible']


@pytest.mark.parametrize('mutation',[
    'DELETE FROM NVTX_EVENTS WHERE start=0',
    'UPDATE NVTX_EVENTS SET end=190 WHERE start=0',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE correlationId=0',
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=0',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=99 WHERE correlationId=0',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=95 WHERE correlationId=0',
])
def test_refuses_ambiguous_missing_or_incomplete_records(capture,mutation):
    path,packet=capture
    with sqlite3.connect(path) as c:c.execute(mutation)
    with pytest.raises(ValueError):attribute(packet,path)


def test_image_change_cannot_reuse_range(capture):
    path,packet=capture
    packet['rows'][0]['image']='f'*64
    with pytest.raises(ValueError):attribute(packet,path)
