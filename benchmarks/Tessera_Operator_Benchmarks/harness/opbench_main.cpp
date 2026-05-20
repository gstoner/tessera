#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <array>
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
               "  --backend reference|artifact|tessera-runtime\n"
               "  --runtime bridge|native    tessera-runtime mode (default bridge)\n"
               "  --artifact-root PATH       Root containing mlir/tessera_ir_samples\n"
               "  --json                     Emit shared benchmark JSON row\n"
               "  --iters N                  Iterations (default 50)\n"
               "  --seed S                   RNG seed (default 123)\n"
               "Matmul args: --m --n --k\n"
               "Conv2d NHWC args: --Nn --H --W --C --Kc --R --S [--stride_h --stride_w --pad_h --pad_w]\n"
               "Attention args: --B --heads --seq --dim\n";
}

static std::string json_escape(const std::string& s){
  std::ostringstream out;
  for(char c: s){
    switch(c){
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

static void emit_json(const std::string& op,
                      const std::string& backend,
                      const std::string& compiler_path,
                      const std::string& runtime_status,
                      const OpResult& res,
                      const OpArgs& args,
                      const std::string& reason,
                      const std::string& artifact_path = ""){
  std::string execution_kind = runtime_status=="artifact_only" ? "artifact_only" :
                               runtime_status=="executable" ? "reference" :
                               "unknown";
  double tflops = res.gflops / 1000.0;
  double bandwidth_gbps = res.gbps;
  std::string telemetry_status = runtime_status=="executable" ? "ok" :
                                 runtime_status=="artifact_only" ? "unmeasured" :
                                 runtime_status=="backend_unavailable" ? "backend_unavailable" :
                                 runtime_status;
  std::cout << "{"
            << "\"operator\":{\"name\":\"" << json_escape(op) << "\",\"dtype\":\"f32\",\"shape\":\"cli\",\"target\":\"cpu\"},"
            << "\"compiler_path\":\"" << compiler_path << "\","
            << "\"runtime_status\":\"" << runtime_status << "\","
            << "\"execution_kind\":\"" << execution_kind << "\","
            << "\"artifact_levels\":{\"graph\":" << ((compiler_path=="artifact_only") ? "true" : "false")
            << ",\"schedule\":false,\"tile\":false,\"target\":false,\"artifact_hash\":null},"
            << "\"correctness\":{\"max_error\":" << res.l2_ref << ",\"relative_error\":null,\"tolerance\":null,\"passed\":null},"
            << "\"profile\":{\"cpu_wall_ms\":" << res.avg_ms << ",\"kernel_elapsed_ms\":null,\"memory_bytes\":null,\"launch_overhead_ms\":null},"
            << "\"metrics\":{\"backend\":\"" << json_escape(backend) << "\",\"gflops\":" << res.gflops
            << ",\"gbps\":" << res.gbps << "},"
            << "\"telemetry\":{\"schema\":\"tessera.telemetry.v1\","
            << "\"name\":\"" << json_escape(op) << "\","
            << "\"source\":\"tessera_operator_bench\","
            << "\"op\":\"" << json_escape(op) << "\","
            << "\"dtype\":\"f32\","
            << "\"arch\":\"cpu\","
            << "\"latency_ms\":" << res.avg_ms << ","
            << "\"tflops\":" << tflops << ","
            << "\"bandwidth_gbps\":" << bandwidth_gbps << ","
            << "\"status\":\"" << telemetry_status << "\","
            << "\"counters\":{},"
            << "\"metadata\":{\"backend\":\"" << json_escape(backend) << "\","
            << "\"compiler_path\":\"" << json_escape(compiler_path) << "\","
            << "\"runtime_status\":\"" << json_escape(runtime_status) << "\","
            << "\"execution_kind\":\"" << json_escape(execution_kind) << "\","
            << "\"artifact_path\":\"" << json_escape(artifact_path) << "\","
            << "\"M\":" << args.M << ",\"N\":" << args.N << ",\"K\":" << args.K
            << ",\"Nn\":" << args.Nn << ",\"H\":" << args.H << ",\"W\":" << args.W
            << ",\"C\":" << args.C << ",\"Kc\":" << args.Kc << ",\"R\":" << args.R << ",\"S\":" << args.S
            << ",\"B\":" << args.B << ",\"heads\":" << args.heads << ",\"seq\":" << args.seq << ",\"dim\":" << args.dim
            << "},\"bottleneck\":\"" << ((res.gflops>0.0) ? "unknown" : (res.gbps>0.0 ? "memory_bound" : "failed_or_unmeasured")) << "\"},"
            << "\"reason\":\"" << json_escape(reason) << "\""
            << "}\n";
}

static std::string shell_quote(const std::string& s){
  std::string out = "'";
  for(char c: s){
    if(c=='\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

static std::string benchmark_bridge_script(const std::string& artifact_root){
  std::string prefix = artifact_root.empty() ? "." : artifact_root;
  if(!prefix.empty() && prefix.back()!='/') prefix += "/";
  return prefix + "scripts/opbench_bridge.py";
}

static int run_python_bridge(const std::string& op,
                             const std::string& backend,
                             const std::string& mode,
                             const std::string& runtime_mode,
                             const std::string& artifact_root,
                             const OpArgs& args){
  const char* python_env = std::getenv("OPBENCH_PYTHON");
  std::string python = python_env && *python_env ? python_env : "python3";
  std::ostringstream cmd;
  cmd << shell_quote(python)
      << " " << shell_quote(benchmark_bridge_script(artifact_root))
      << " --mode " << shell_quote(mode)
      << " --backend " << shell_quote(backend)
      << " --runtime " << shell_quote(runtime_mode)
      << " --op " << shell_quote(op)
      << " --iters " << args.iters
      << " --seed " << args.seed
      << " --m " << args.M
      << " --n " << args.N
      << " --k " << args.K
      << " --Nn " << args.Nn
      << " --H " << args.H
      << " --W " << args.W
      << " --C " << args.C
      << " --Kc " << args.Kc
      << " --R " << args.R
      << " --S " << args.S
      << " --stride_h " << args.stride_h
      << " --stride_w " << args.stride_w
      << " --pad_h " << args.pad_h
      << " --pad_w " << args.pad_w
      << " --B " << args.B
      << " --heads " << args.heads
      << " --seq " << args.seq
      << " --dim " << args.dim;

  std::array<char, 4096> buffer{};
  FILE* pipe = popen(cmd.str().c_str(), "r");
  if(!pipe){
    OpResult res{};
    emit_json(op, backend, "runtime_unavailable", "backend_unavailable", res, args, "failed to start Python Tessera runtime bridge");
    return 2;
  }
  while(fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr){
    std::cout << buffer.data();
  }
  int rc = pclose(pipe);
  return rc == 0 ? 0 : 2;
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
  std::string backend = "reference";
  std::string runtime_mode = "bridge";
  std::string artifact_root = ".";
  bool json = false;
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
    else if(a=="--backend" && i+1<argc){ backend = argv[++i]; }
    else if(a=="--runtime" && i+1<argc){ runtime_mode = argv[++i]; }
    else if(a=="--artifact-root" && i+1<argc){ artifact_root = argv[++i]; }
    else if(a=="--json"){ json = true; }
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

  if(backend=="artifact"){
    if(json) return run_python_bridge(op, backend, "artifact", runtime_mode, artifact_root, args);
    std::cout << "artifact_mode=generated_bundle bridge=" << benchmark_bridge_script(artifact_root) << "\n";
    return 0;
  }

  if(backend=="tessera-runtime"){
    if(runtime_mode=="native"){
      OpResult res{};
      std::string reason = "Native C ABI operator launch is pending; use --runtime bridge";
      if(json) emit_json(op, backend, "runtime_unavailable", "backend_unavailable", res, args, reason);
      else std::cout << "runtime_status=backend_unavailable reason=" << reason << "\n";
      return 0;
    }
    if(runtime_mode!="bridge"){
      std::cerr<<"Unknown runtime: "<<runtime_mode<<"\n"; usage(); return 1;
    }
    if(json) return run_python_bridge(op, backend, "runtime", runtime_mode, artifact_root, args);
    OpResult res{};
    std::string reason = "Python Tessera runtime bridge requires --json output in this harness";
    if(json) emit_json(op, backend, "runtime_unavailable", "backend_unavailable", res, args, reason);
    else std::cout << "runtime_status=backend_unavailable reason=" << reason << "\n";
    return 0;
  }

  if(backend!="reference"){
    std::cerr<<"Unknown backend: "<<backend<<"\n"; usage(); return 1;
  }

  opbench_device_init();
  NvtxRange R(("opbench:"+op).c_str());
  auto res = fn(args);
  if(json) emit_json(op, backend, "reference", "executable", res, args, "");
  else std::cout<<"avg_ms="<<res.avg_ms<<" gflops="<<res.gflops<<" gbps="<<res.gbps<<" l2_ref="<<res.l2_ref<<"\n";
  return 0;
}
