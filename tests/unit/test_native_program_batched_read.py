"""Execute production native readback bodies with deferred-copy failure shims."""
import ctypes as c
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from tessera.compiler.native_scaled_program import PreparedScaledProgram

@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler=shutil.which("g++") or shutil.which("clang++")
    if not compiler: pytest.skip("native C++ compiler required")
    root=Path(__file__).resolve().parents[2]
    source=(root/"src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp").read_text()
    begin=source.index('extern "C" int tessera_rocm_program_read(')
    end=source.index('extern "C" int tessera_rocm_program_close(',begin)
    prefix=r'''
#include <cstdint>
#include <cstring>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <unistd.h>
struct Contract { uint64_t bytes; uint32_t ownership; };
struct Program {
  std::vector<Contract> contract;
  std::vector<void *> buffers;
  bool poisoned=false,output=true,pinned=false;
  uint64_t generation=3,pinnedReadbackBytes=48;
  std::vector<unsigned char> readback;
  void *pinnedReadback=nullptr,*stream=nullptr;
};
struct State { std::mutex mutex; std::map<uint64_t,std::unique_ptr<Program>> programs; };
State pool;
const pid_t process=getpid();
State &state(){return pool;}
bool sameIdentity=true;
bool identity(const Program &){return sameIdentity;}
int copies=0,syncs=0,failCopy=0,failSync=0;
const int hipSuccess=0,hipMemcpyDeviceToHost=2;
struct Copy{void *destination; const void *source; uint64_t bytes;};
std::vector<Copy> queued;
float data[4][4];
unsigned char pinnedSlab[48];
int hipMemcpyAsync(void *dst,const void *src,uint64_t bytes,int,void *){
  ++copies;if(failCopy==copies)return 1;
  queued.push_back({dst,src,bytes});return 0;
}
int hipStreamSynchronize(void *){
  ++syncs;if(failSync)return 1;
  for(auto &copy:queued)std::memcpy(copy.destination,copy.source,copy.bytes);
  queued.clear();return 0;
}
extern "C" void setup(int pinned){
  queued.clear();pool.programs.clear();sameIdentity=true;
  copies=syncs=failCopy=failSync=0;
  auto p=std::make_unique<Program>();p->pinned=pinned;
  p->pinnedReadback=pinnedSlab;
  for(unsigned i=0;i<4;++i){
    p->contract.push_back({16,i?2U:0U});
    for(unsigned j=0;j<4;++j)data[i][j]=float(i*10+j);
    p->buffers.push_back(data[i]);
  }
  pool.programs.emplace(7,std::move(p));
}
extern "C" void fail(int copy,int sync){failCopy=copy;failSync=sync;}
extern "C" void identity_failure(){sameIdentity=false;}
extern "C" int stats(int kind){return kind==0?copies:kind==1?syncs:pool.programs.at(7)->poisoned;}
'''
    folder=tmp_path_factory.mktemp("native-readback")
    cpp=folder/"read.cpp";so=folder/"read.so"
    cpp.write_text(prefix+source[begin:end])
    subprocess.run([compiler,"-std=c++17","-shared","-fPIC",str(cpp),"-o",str(so)],check=True)
    lib=c.CDLL(str(so))
    lib.tessera_rocm_program_read_many.argtypes=[c.c_uint64,c.c_uint64,c.c_uint32,c.POINTER(c.c_uint32),c.POINTER(c.c_void_p),c.POINTER(c.c_uint64)]
    lib.tessera_rocm_program_read.argtypes=[c.c_uint64,c.c_uint32,c.c_uint64,c.c_void_p,c.c_uint64]
    lib.stats.restype=c.c_int
    return lib

def owner_for(lib):
    owner=PreparedScaledProgram.__new__(PreparedScaledProgram)
    owner.handle=c.c_uint64(7);owner.lib=lib
    owner._read_many=lib.tessera_rocm_program_read_many
    owner._binding=SimpleNamespace(outputs=(1,2,3),storage=[
        SimpleNamespace(shape=(4,),storage="f32") for _ in range(4)])
    return owner

@pytest.mark.parametrize("pinned",[0,1])
@pytest.mark.parametrize("batched",[False,True])
def test_native_read_publishes_correct_independent_outputs_after_completion(library,pinned,batched):
    library.setup(pinned)
    outputs=owner_for(library).read(3,batched=batched)
    for slot,output in enumerate(outputs,1):
        np.testing.assert_array_equal(output,np.arange(4,dtype=np.float32)+slot*10)
    assert library.stats(0)==3
    assert library.stats(1)==(1 if batched else 3)
    assert all(not np.shares_memory(a,b) for i,a in enumerate(outputs) for b in outputs[i+1:])

@pytest.mark.parametrize("failure",["copy","completion"])
@pytest.mark.parametrize("pinned",[0,1])
def test_failed_native_batch_never_publishes_partial_caller_outputs(library,pinned,failure):
    library.setup(pinned);library.fail(2 if failure=="copy" else 0,failure=="completion")
    outputs=[np.full(4,-99,np.float32) for _ in range(3)]
    slots=(c.c_uint32*3)(1,2,3);sizes=(c.c_uint64*3)(16,16,16)
    pointers=(c.c_void_p*3)(*(output.ctypes.data for output in outputs))
    read=library.tessera_rocm_program_read_many
    assert read(7,3,3,slots,pointers,sizes)==(5 if failure=="copy" else 7)
    for output in outputs:np.testing.assert_array_equal(output,-99)
    assert library.stats(2)==1
    assert read(7,3,3,slots,pointers,sizes)==10

@pytest.mark.parametrize("mutation",["count","slot","private","bytes","duplicate","overlap","overflow","generation","identity"])
def test_native_batch_refuses_complete_frame_before_enqueue(library,mutation):
    library.setup(0)
    outputs=[np.full(4,-99,np.float32) for _ in range(3)]
    slots=[1,2,3];sizes=[16]*3;pointers=[out.ctypes.data for out in outputs]
    count=3;generation=3
    if mutation=="count":count=0
    elif mutation=="slot":slots[2]=4
    elif mutation=="private":slots[2]=0
    elif mutation=="bytes":sizes[2]=15
    elif mutation=="duplicate":slots[2]=slots[1]
    elif mutation=="overlap":pointers[2]=pointers[1]+4
    elif mutation=="overflow":pointers[2]=2**(c.sizeof(c.c_void_p)*8)-8
    elif mutation=="generation":generation=2
    elif mutation=="identity":library.identity_failure()
    code=library.tessera_rocm_program_read_many(7,generation,count,
        (c.c_uint32*3)(*slots),(c.c_void_p*3)(*pointers),(c.c_uint64*3)(*sizes))
    assert code==(10 if mutation=="generation" else 2 if mutation=="identity" else 1)
    assert library.stats(0)==0 and library.stats(1)==0
    for output in outputs:np.testing.assert_array_equal(output,-99)

def test_reordered_subset_uses_checked_returned_slots(library):
    library.setup(0)
    outputs=[np.empty(4,np.float32) for _ in range(2)]
    assert library.tessera_rocm_program_read_many(7,3,2,(c.c_uint32*2)(3,1),
        (c.c_void_p*2)(*(out.ctypes.data for out in outputs)),(c.c_uint64*2)(16,16))==0
    np.testing.assert_array_equal(outputs[0],np.arange(4)+30)
    np.testing.assert_array_equal(outputs[1],np.arange(4)+10)
    assert library.stats(1)==1
