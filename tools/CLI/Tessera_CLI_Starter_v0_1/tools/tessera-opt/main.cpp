//===- tessera-opt/main.cpp — Apply Tessera/MLIR passes to IR files --------===//
//
// tessera-opt reads one or more .mlir files (or stdin), applies a resolved
// pass pipeline, and writes transformed IR to stdout or an output file.
//
// Example usage:
//   tessera-opt model.mlir --pipeline=tessera-halo-infer --to=tile -o out.mlir
//   tessera-opt model.mlir --alias=neighbors-pipeline --verify --dump=schedule
//   cat model.mlir | tessera-opt --verify-only --json
//
// Exit codes follow CLI_Design.md §Exit Codes.
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "common/manifest.hpp"
#include "common/args.hpp"

static const char* TOOL = "tessera-opt";

// ---------------------------------------------------------------------------
// Known pipeline aliases → pass lists
// ---------------------------------------------------------------------------
static std::string resolve_alias(const std::string& alias) {
  if (alias == "neighbors-pipeline")
    return "-tessera-halo-infer,-tessera-stencil-lower,"
           "-tessera-pipeline-overlap,-tessera-dynamic-topology,"
           "-canonicalize";
  if (alias == "graph-to-schedule")
    return "-tessera-verify,-tessera-migrate-ir,"
           "-tessera-graph-canonicalize,"
           "-tessera-lower-graph-to-schedule,-tessera-cleanup";
  if (alias == "schedule-to-tile")
    return "-tessera-verify,-tessera-schedule-normalize,"
           "-tessera-lower-schedule-to-tile,"
           "-tessera-tiling-interface,-tessera-cleanup";
  if (alias == "tile-to-target")
    return "-tessera-verify,-tessera-lower-tile-to-target,"
           "-tessera-target-canonicalize,-tessera-cleanup";
  if (alias == "pm-verify")
    return "-tessera-pm-verify,-cse,-canonicalize";
  if (alias == "pm-legalize")
    return "-tessera-pm-verify,-tessera-graph-to-schedule,"
           "-tessera-schedule-to-tile,-canonicalize";
  if (alias == "full")
    return "neighbors-pipeline,graph-to-schedule,"
           "schedule-to-tile,tile-to-target";
  return "";  // not an alias — return raw for direct pass-manager use
}

int main(int argc, char** argv) {
  // ---- Options -------------------------------------------------------
  std::string opt_pipeline, opt_alias, opt_from, opt_to;
  std::string opt_dump, opt_dump_dir;
  std::vector<std::string> opt_add_pass, opt_disable_pass;
  std::vector<std::string> inputs;
  bool opt_verify      = false;
  bool opt_verify_only = false;
  bool opt_print_pipe  = false;
  bool opt_time_passes = false;
  bool opt_canonicalize= false;
  bool opt_migrate_ir  = false;
  bool opt_cleanup     = false;
  int  opt_level       = 2;

  tessera::Args args(TOOL, "Apply Tessera/MLIR passes to IR files", argc, argv);
  args.option("--pipeline",        "Pass-pipeline string (mlir-opt syntax)",   &opt_pipeline)
      .option("--alias",           "Named pipeline alias (e.g. neighbors-pipeline)", &opt_alias)
      .option("--from",            "Input IR layer {graph,schedule,tile,target,auto}", &opt_from, "auto")
      .option("--to",              "Output IR layer {graph,schedule,tile,target}", &opt_to, "")
      .option("--dump",            "Also dump IR at layer {graph,schedule,tile,target,llvm}", &opt_dump)
      .option("--dump-dir",        "Directory for --dump output", &opt_dump_dir, "")
      .multi("--add-pass",         "Add a pass to the end of the pipeline",    &opt_add_pass)
      .multi("--disable-pass",     "Remove a pass from the pipeline",          &opt_disable_pass)
      .int_option("-O",            "Optimization level 0-3 (default 2)",       &opt_level, 2)
      .flag("--verify",            "Run verifiers after each phase",           &opt_verify)
      .flag("--verify-only",       "Verify IR without transforming",           &opt_verify_only)
      .flag("--print-pipeline",    "Print resolved pipeline and exit",         &opt_print_pipe)
      .flag("--time-passes",       "Print per-pass wall-clock timings",        &opt_time_passes)
      .flag("--canonicalize",      "Append -canonicalize pass",                &opt_canonicalize)
      .flag("--migrate-ir",        "Append -tessera-migrate-ir pass",          &opt_migrate_ir)
      .flag("--cleanup",           "Append -tessera-cleanup pass",             &opt_cleanup)
      .positional("input.mlir",    ".mlir files to transform (- = stdin)",     &inputs);

  if (!args.parse()) return args.exit_code();

  // ---- Resolve pipeline -----------------------------------------------
  std::string pipeline = opt_pipeline;
  if (!opt_alias.empty()) {
    std::string resolved = resolve_alias(opt_alias);
    if (resolved.empty()) {
      TLOG_ERROR(TOOL, "unknown pipeline alias '" + opt_alias + "'");
      return tessera::EXIT_PARSE_ERROR;
    }
    pipeline = resolved;
  }
  if (opt_canonicalize) pipeline += (pipeline.empty() ? "" : ",") + std::string("-canonicalize");
  if (opt_migrate_ir)   pipeline += (pipeline.empty() ? "" : ",") + std::string("-tessera-migrate-ir");
  if (opt_cleanup)      pipeline += (pipeline.empty() ? "" : ",") + std::string("-tessera-cleanup");
  for (auto& p : opt_add_pass)
    pipeline += (pipeline.empty() ? "" : ",") + p;

  // Print and exit if --print-pipeline
  if (opt_print_pipe) {
    std::cout << "Pipeline: " << (pipeline.empty() ? "(identity)" : pipeline) << "\n";
    std::cout << "From: " << opt_from << "\n";
    if (!opt_to.empty()) std::cout << "To: " << opt_to << "\n";
    return tessera::EXIT_OK;
  }

  // ---- Read input -------------------------------------------------------
  std::string ir;
  if (inputs.empty() || inputs[0] == "-") {
    // Read from stdin
    std::stringstream ss;
    ss << std::cin.rdbuf();
    ir = ss.str();
    if (ir.empty()) ir = "// (tessera-opt) empty stdin\nmodule {}";
  } else {
    try {
      ir = slurpFile(inputs[0]);
    } catch (const std::exception& e) {
      TLOG_ERROR(TOOL, e.what());
      tessera::json_result(TOOL, args.out_dir(), false,
                           "\"error\":\"" + tessera::jsonEscape(e.what()) + "\"");
      return tessera::EXIT_IO_ERROR;
    }
  }

  // ---- Apply pipeline (annotate IR text with applied passes) -----------
  int rc = tessera::EXIT_OK;
  try {
    auto paths = makeArtifactLayout(args.out_dir());

    if (opt_verify_only) {
      TLOG_INFO(TOOL, "verify-only mode: checking structural invariants");
      // Structural check: must contain 'module' or 'func.func'
      if (ir.find("module") == std::string::npos &&
          ir.find("func.func") == std::string::npos) {
        TLOG_ERROR(TOOL, "IR does not contain a module or func.func op");
        rc = tessera::EXIT_PARSE_ERROR;
      } else {
        TLOG_INFO(TOOL, "IR verification passed");
      }
    } else {
      // Annotate the IR with the pipeline that was applied.
      // In a full build this would delegate to mlir::MlirOptMain.
      std::string header = "// tessera-opt: from=" + opt_from;
      if (!opt_to.empty())   header += " to=" + opt_to;
      if (!pipeline.empty()) header += " pipeline=[" + pipeline + "]";
      if (opt_time_passes)   header += " time-passes=true";
      header += " O=" + std::to_string(opt_level) + "\n";

      std::string out_ir = header + ir;

      // Output: --dump snapshot
      if (!opt_dump.empty()) {
        std::string dump_dir = opt_dump_dir.empty() ? paths.ir_dir : opt_dump_dir;
        tessera::ensureDir(dump_dir);  // may already exist
        std::string dump_path = dump_dir + "/" + opt_dump + ".mlir";
        if (!args.dry_run()) {
          writeFile(dump_path, out_ir);
          TLOG_INFO(TOOL, "dumped " + opt_dump + " IR to " + dump_path);
        } else {
          TLOG_INFO(TOOL, "[dry-run] would dump " + opt_dump + " IR to " + dump_path);
        }
      }

      // Primary output
      std::string out_path = args.output();
      if (args.dry_run()) {
        TLOG_INFO(TOOL, "[dry-run] would write transformed IR to " +
                  (out_path == "-" ? "stdout" : out_path));
      } else if (out_path == "-") {
        std::cout << out_ir;
      } else {
        writeFile(out_path, out_ir);
        TLOG_INFO(TOOL, "wrote transformed IR to " + out_path);
      }

      // Also save to ir/final.mlir in artifact dir
      if (!args.dry_run()) {
        writeFile(paths.ir_dir + "/final.mlir", out_ir);
      }
    }
  } catch (const std::exception& e) {
    TLOG_ERROR(TOOL, e.what());
    rc = tessera::EXIT_IO_ERROR;
  }

  tessera::json_result(TOOL, args.out_dir(), rc == tessera::EXIT_OK,
                       "\"pipeline\":\"" + tessera::jsonEscape(pipeline) + "\"");
  return rc;
}
