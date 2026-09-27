# TesseraLit.cmake -- one validated lit runner for every lit suite in the tree.
#
# Why this exists (SMALL-CORRECTNESS-GAPS-2026-09-27).  Seven CMake sites each
# ran their own `find_program(... llvm-lit lit.py lit)` and trusted whatever
# file they found.  On Tajasarus both LLVM 23.1.1 prefixes ship a lit *wrapper
# script* whose `import lit` points at a source tree that is not on the box
# (`/usr/lib/llvm-23/utils/lit`, `../../llvm-project-23.1.1-assertions/...`),
# so the selected runner died with `ModuleNotFoundError: No module named 'lit'`
# and `ninja check-tessera-ir` ran zero fixtures.  A file that exists is not a
# runner that works: every candidate is now executed with `--version` and only
# a candidate that answers is selected.  A rejected candidate is reported with
# its own error line, so the failure cannot be mistaken for "no lit installed".
#
# Selection order (first working candidate wins):
#   1. -DTESSERA_LIT=<path>         explicit override (validated like the rest;
#                                   a stale cached value from an older
#                                   configure is indistinguishable, so a broken
#                                   one warns and falls through rather than
#                                   stopping configure)
#   2. -DLLVM_EXTERNAL_LIT=<path>   the LLVM-standard override (release gates
#                                   pass it)
#   3. <repo>/.venv/bin             the project venv, which non-interactive
#                                   configures do not activate
#   4. $VIRTUAL_ENV/bin             an activated venv
#   5. LLVM_TOOLS_BINARY_DIR        the matched LLVM's own llvm-lit / lit
#   6. every $PATH entry, in order
#
# Result: global property TESSERA_LIT_COMMAND (empty when nothing works) and
# the cache entry TESSERA_LIT_SELECTED (INTERNAL, for inspection).  Callers
# use tessera_lit_command(<var>) and decide their own policy for "none":
# tests/ fails configure; backend suites define a check target that FAILS
# with the rejection list instead of echoing "skipping" and exiting 0.
include_guard(GLOBAL)

function(_tessera_lit_probe candidate out_ok out_detail)
  execute_process(
    COMMAND "${candidate}" --version
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    TIMEOUT 60)
  set(_all "${_out}\n${_err}")
  if(_rc STREQUAL "0" AND _all MATCHES "lit[ \t]+[0-9]")
    set(${out_ok} TRUE PARENT_SCOPE)
    string(STRIP "${_out}${_err}" _version)
    set(${out_detail} "${_version}" PARENT_SCOPE)
    return()
  endif()
  # Report the last non-empty line: for a broken wrapper that is the
  # exception ("ModuleNotFoundError: No module named 'lit'").
  string(STRIP "${_err}" _err)
  if(_err STREQUAL "")
    string(STRIP "${_out}" _err)
  endif()
  string(REPLACE "\n" ";" _lines "${_err}")
  set(_last "exit status ${_rc}")
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    if(NOT _line STREQUAL "")
      set(_last "${_line} (exit status ${_rc})")
    endif()
  endforeach()
  set(${out_ok} FALSE PARENT_SCOPE)
  set(${out_detail} "${_last}" PARENT_SCOPE)
endfunction()

function(tessera_resolve_lit)
  get_property(_done GLOBAL PROPERTY TESSERA_LIT_RESOLVED SET)
  if(_done)
    return()
  endif()

  get_filename_component(_repo "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  set(_candidates "")
  foreach(_explicit IN ITEMS "${TESSERA_LIT}" "${LLVM_EXTERNAL_LIT}")
    if(NOT _explicit STREQUAL "" AND NOT _explicit MATCHES "-NOTFOUND$")
      list(APPEND _candidates "${_explicit}")
    endif()
  endforeach()
  set(_dirs "${_repo}/.venv/bin")
  if(DEFINED ENV{VIRTUAL_ENV} AND NOT "$ENV{VIRTUAL_ENV}" STREQUAL "")
    list(APPEND _dirs "$ENV{VIRTUAL_ENV}/bin")
  endif()
  if(LLVM_TOOLS_BINARY_DIR)
    list(APPEND _dirs "${LLVM_TOOLS_BINARY_DIR}")
  endif()
  if(CMAKE_HOST_WIN32)
    set(_path_list "$ENV{PATH}")
  else()
    string(REPLACE ":" ";" _path_list "$ENV{PATH}")
  endif()
  list(APPEND _dirs ${_path_list})
  foreach(_dir IN LISTS _dirs)
    if(_dir STREQUAL "")
      continue()
    endif()
    foreach(_name IN ITEMS lit llvm-lit lit.py)
      list(APPEND _candidates "${_dir}/${_name}")
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _candidates)

  set(_selected "")
  set(_rejected "")
  foreach(_candidate IN LISTS _candidates)
    if(NOT EXISTS "${_candidate}" OR IS_DIRECTORY "${_candidate}")
      continue()
    endif()
    _tessera_lit_probe("${_candidate}" _ok _detail)
    if(_ok)
      set(_selected "${_candidate}")
      message(STATUS "Tessera lit runner: ${_candidate} (${_detail})")
      break()
    endif()
    list(APPEND _rejected "${_candidate}: ${_detail}")
    message(WARNING
      "Tessera: rejected lit runner ${_candidate} -- it exists but cannot "
      "run: ${_detail}. Continuing the search; pass -DTESSERA_LIT=<working "
      "lit> to choose one explicitly.")
  endforeach()

  if(_selected STREQUAL "")
    list(LENGTH _rejected _nrej)
    message(WARNING
      "Tessera: NO working lit runner found (${_nrej} candidate(s) rejected). "
      "Every lit check target will FAIL with this message rather than skip. "
      "Install lit (`pip install lit`) or pass -DTESSERA_LIT=<path>.")
  endif()

  set_property(GLOBAL PROPERTY TESSERA_LIT_RESOLVED TRUE)
  set_property(GLOBAL PROPERTY TESSERA_LIT_COMMAND "${_selected}")
  set_property(GLOBAL PROPERTY TESSERA_LIT_REJECTED "${_rejected}")
  set(TESSERA_LIT_SELECTED "${_selected}" CACHE INTERNAL
      "Validated lit runner chosen by cmake/TesseraLit.cmake")
endfunction()

# tessera_lit_command(<out-var>) -- the validated runner, or "" when none works.
function(tessera_lit_command out_var)
  tessera_resolve_lit()
  get_property(_cmd GLOBAL PROPERTY TESSERA_LIT_COMMAND)
  set(${out_var} "${_cmd}" PARENT_SCOPE)
endfunction()

# tessera_add_lit_unavailable_target(<target> <suite description>) -- the
# check target for a suite that has no working runner.  It fails, naming every
# rejected candidate, so a build that "ran the lit suite" can never exit 0
# having run nothing.
function(tessera_add_lit_unavailable_target target suite)
  tessera_resolve_lit()
  get_property(_rejected GLOBAL PROPERTY TESSERA_LIT_REJECTED)
  set(_why "none found")
  if(_rejected)
    string(REPLACE ";" " | " _why "${_rejected}")
  endif()
  add_custom_target(${target}
    COMMAND ${CMAKE_COMMAND} -E echo
            "${target}: cannot run ${suite} -- no working lit runner (${_why}). Reconfigure with -DTESSERA_LIT=<working lit>."
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)
endfunction()
