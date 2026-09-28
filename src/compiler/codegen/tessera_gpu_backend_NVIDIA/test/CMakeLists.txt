# Shared validated lit runner (cmake/TesseraLit.cmake); honors an explicit
# -DLLVM_EXTERNAL_LIT (the release gate passes one) after checking it runs.
include(${CMAKE_CURRENT_LIST_DIR}/../../../../../cmake/TesseraLit.cmake)
tessera_lit_command(_tessera_nvidia_lit)

if(NOT TESSERA_LLVM_TOOLS_DIR)
  if(LLVM_TOOLS_BINARY_DIR)
    set(TESSERA_LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
  else()
    find_program(TESSERA_FILECHECK NAMES FileCheck)
    if(TESSERA_FILECHECK)
      get_filename_component(TESSERA_LLVM_TOOLS_DIR "${TESSERA_FILECHECK}" DIRECTORY)
    endif()
  endif()
endif()

set(TESSERA_NVIDIA_OPT_EXECUTABLE
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/tools/tessera-nvidia-opt)
configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

if(_tessera_nvidia_lit)
  add_custom_target(check-tessera-nvidia
    COMMAND ${_tessera_nvidia_lit} -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tessera-nvidia-opt)
else()
  tessera_add_lit_unavailable_target(check-tessera-nvidia "the NVIDIA backend lit suite")
  add_dependencies(check-tessera-nvidia tessera-nvidia-opt)
endif()
