# The tessera_x86 Target IR dialect is HARDWARE-FREE by contract (Decision #19):
# its whole reason to exist is that lit can check it "on any host, including
# hosts with no AMX". A host-specific ISA flag reaching this translation unit
# breaks that contract in the worst possible way -- not at build time, but as a
# SIGILL the first time the dialect registers itself, which reads as a corrupt
# MLIR install rather than a build misconfiguration
# (`X86-DIALECT-LOAD-CRASH-2026-08-12`).
#
# Fail the CONFIGURE instead. Per Decision #21a this is a semantic key, so it
# fails closed and says why rather than quietly building something that only
# runs on some machines.
get_directory_property(_tessera_x86_ir_dir_opts COMPILE_OPTIONS)
if(_tessera_x86_ir_dir_opts MATCHES "-mavx|-mamx|-march=|-mtune=")
  message(FATAL_ERROR
    "tessera_x86 Target IR dialect would be compiled with host-specific ISA "
    "flags (${_tessera_x86_ir_dir_opts}). The dialect must stay baseline so it "
    "loads on hosts without AVX-512/AMX. Apply arch flags with "
    "target_compile_options() on the kernel targets, never add_compile_options() "
    "in a directory that has subdirectories.")
endif()

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraX86/IR/TesseraX86Ops.td)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/TesseraX86/IR)
mlir_tablegen(TesseraX86Ops.h.inc -gen-op-decls)
mlir_tablegen(TesseraX86Ops.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraX86Dialect.h.inc -gen-dialect-decls -dialect=tessera_x86)
mlir_tablegen(TesseraX86Dialect.cpp.inc -gen-dialect-defs -dialect=tessera_x86)
mlir_tablegen(TesseraX86Types.h.inc -gen-typedef-decls -typedefs-dialect=tessera_x86)
mlir_tablegen(TesseraX86Types.cpp.inc -gen-typedef-defs -typedefs-dialect=tessera_x86)
add_public_tablegen_target(TesseraX86TableGen)

add_mlir_library(TesseraX86IR
  TesseraX86Ops.cpp
  LINK_LIBS PUBLIC
    MLIRIR MLIRFuncDialect MLIRMemRefDialect
  DEPENDS TesseraX86TableGen)
target_include_directories(TesseraX86IR PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR})
