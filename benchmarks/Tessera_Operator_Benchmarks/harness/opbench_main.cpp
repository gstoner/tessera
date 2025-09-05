#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include "common/timer.h"
#include "common/nvtx_helpers.h"
#include "common/device_utils.h"
#include "harness/op_registry.h"

// Forward-declare op registration from each file
void register_matmul();
void register_conv2d();
void register_flash_attention();
void register_reduce();
void register_elementwise();
void register_softmax_layernorm();
void register_transpose_gather();

static void usage(){
  std::cout << "opbench --op <name> [args]\n"
               "  --list-ops                 List available operators\n"
               "  --iters N                  Iterations (default 50)\n"
               "  --seed S                   RNG seed (default 123)\n"
               "Matmul args: --m --n --k\n"
               "Conv2d NHWC args: --Nn --H --W --C --Kc --R --S [--stride_h --stride_w --pad_h --pad_w]\n"
               "Attention args: --B --heads --seq --dim\n";
}

int main(int argc, char** argv){
  register_matmul();
  register_conv2d();
  register_flash_attention();
  register_reduce();
  register_elementwise();
  register_softmax_layernorm();
  register_transpose_gather();

  if(argc==1){ usage(); return 0; }

  std::string op;
  OpArgs args;
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    auto want_int=[&](int& ref){
      if(i+1>=argc){ std::cerr<<"missing value for "<<a<<"\n"; return; }
      ref = std::atoi(argv[++i]);
    };
    if(a=="--list-ops"){
      for(auto& info: OpRegistry::instance().list()) std::cout<<info.name<<"\n";
      return 0;
    } else if(a=="--op" && i+1<argc){ op = argv[++i]; }
    else if(a=="--iters"){ want_int(args.iters); }
    else if(a=="--seed"){ if(i+1<argc) args.seed = std::strtoull(argv[++i],nullptr,10); }
    else if(a=="--m"){ want_int(args.M); }
    else if(a=="--n"){ want_int(args.N); }
    else if(a=="--k"){ want_int(args.K); }
    else if(a=="--Nn"){ want_int(args.Nn); }
    else if(a=="--H"){ want_int(args.H); }
    else if(a=="--W"){ want_int(args.W); }
    else if(a=="--C"){ want_int(args.C); }
    else if(a=="--Kc"){ want_int(args.Kc); }
    else if(a=="--R"){ want_int(args.R); }
    else if(a=="--S"){ want_int(args.S); }
    else if(a=="--stride_h"){ want_int(args.stride_h); }
    else if(a=="--stride_w"){ want_int(args.stride_w); }
    else if(a=="--pad_h"){ want_int(args.pad_h); }
    else if(a=="--pad_w"){ want_int(args.pad_w); }
    else if(a=="--B"){ want_int(args.B); }
    else if(a=="--heads"){ want_int(args.heads); }
    else if(a=="--seq"){ want_int(args.seq); }
    else if(a=="--dim"){ want_int(args.dim); }
    else {
      std::cerr<<"Unknown arg: "<<a<<"\n"; usage(); return 1;
    }
  }

  if(op.empty()){ std::cerr<<"--op is required\n"; return 1; }
  auto fn = OpRegistry::instance().find(op);
  if(!fn){ std::cerr<<"Unknown op: "<<op<<"\n"; return 1; }

  opbench_device_init();
  NvtxRange R(("opbench:"+op).c_str());
  auto res = fn(args);
  std::cout<<"avg_ms="<<res.avg_ms<<" gflops="<<res.gflops<<" gbps="<<res.gbps<<" l2_ref="<<res.l2_ref<<"\n";
  return 0;
}
