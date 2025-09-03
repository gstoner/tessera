#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

#ifdef TESSERA_HAVE_PJRT_C_API
// Expect pjrt_c_api.h in the include path; provided by OpenXLA
#include "pjrt_c_api.h"
#endif

static std::string slurp(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
#ifndef TESSERA_HAVE_PJRT_C_API
  std::cout << "PJRT TPU Runner (starter)\n";
  const char* dev = std::getenv("PJRT_DEVICE");
  std::cout << "PJRT_DEVICE=" << (dev ? dev : "(unset)") << "\n";
  std::cout << "Rebuild with -DTESSERA_HAVE_PJRT_C_API=ON and provide pjrt_c_api.h to enable compile+execute.\n";
  std::cout << "Usage (then): pjrt_runner --program examples/matmul_128.mlir --format=stablehlo\n";
  return 0;
#else
  std::string program_path = "examples/matmul_128.mlir";
  std::string program_format = "stablehlo";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--program" && i+1 < argc) { program_path = argv[++i]; }
    else if (a == "--format" && i+1 < argc) { program_format = argv[++i]; }
  }

  std::string prog = slurp(program_path);
  if (prog.empty()) {
    std::cerr << "Failed to read program: " << program_path << "\n";
    return 1;
  }

  // 1) Create client for TPU
  PJRT_Client_Create_Args cargs;
  memset(&cargs, 0, sizeof(cargs));
  cargs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  cargs.priv = nullptr;
  cargs.create_options = nullptr;
  PJRT_Client* client = nullptr;
  PJRT_Error* err = PJRT_Client_Create(&cargs, &client);
  if (err) { std::cerr << "PJRT_Client_Create failed\n"; return 1; }

  // 2) Enumerate devices (optional)
  PJRT_Client_Devices_Args dargs;
  memset(&dargs, 0, sizeof(dargs));
  dargs.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  dargs.client = client;
  PJRT_Devices* dv = nullptr;
  err = PJRT_Client_Devices(&dargs, &dv);
  if (err) { std::cerr << "PJRT_Client_Devices failed\n"; return 1; }
  std::cout << "Devices enumerated.\n";

  // 3) Compile program
  PJRT_Program program;
  memset(&program, 0, sizeof(program));
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = prog.data();
  program.code_size = prog.size();
  program.format = program_format.c_str();

  PJRT_Compile_Args pargs;
  memset(&pargs, 0, sizeof(pargs));
  pargs.struct_size = PJRT_Compile_Args_STRUCT_SIZE;
  pargs.client = client;
  pargs.program = &program;
  PJRT_LoadedExecutable* exec = nullptr;
  err = PJRT_Compile(&pargs, &exec);
  if (err) { std::cerr << "PJRT_Compile failed\n"; return 1; }

  // 4) Prepare dummy buffers (zeros) and execute once.
  // NOTE: For a real run, you should query exec->parameter shapes and allocate PJRT buffers.
  std::cout << "Compiled. (Buffer creation/execution elided in starter.)\n";
  std::cout << "TODO: Create PJRT_Buffer for each parameter and call PJRT_Executable_Execute.\n";

  PJRT_Client_Destroy(client);
  return 0;
#endif
}
