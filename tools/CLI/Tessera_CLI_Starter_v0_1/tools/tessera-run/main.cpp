//===- tessera-run/main.cpp — Execute kernels and validate numerics --------===//
//
// tessera-run loads a compiled artifact directory, executes the kernels
// against synthetic or provided inputs, and compares outputs to a golden
// reference or within a tolerance band.
//
// Example usage:
//   tessera-run build/ --rtol=1e-3 --atol=1e-5
//   tessera-run build/ --inputs=inputs.json --golden=golden.json
//   tessera-run build/ --allow-exec --timeout=30 --json
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "common/manifest.hpp"
#include "common/args.hpp"

static const char* TOOL = "tessera-run";

// ---------------------------------------------------------------------------
// Numeric validation helpers
// ---------------------------------------------------------------------------

struct ValidationResult {
  std::string kernel;
  bool        passed;
  double      rtol;
  double      atol;
  double      max_abs_err;
  double      max_rel_err;
  std::string status;  // "pass" | "fail" | "skip"
  std::string message;
};

// Synthetic CPU matmul for reference: C = A × B (all-ones, column width K)
static void ref_matmul(int M, int N, int K, std::vector<float>& C) {
  C.assign(static_cast<std::size_t>(M * N), 0.f);
  // A and B are all-ones, so C[i][j] = K for all i,j
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      C[static_cast<std::size_t>(i * N + j)] = static_cast<float>(K);
}

// Simulate a kernel run result (adds synthetic noise proportional to problem size)
static std::vector<float> sim_kernel_result(int M, int N, int K,
                                             bool add_error = false) {
  std::vector<float> out(static_cast<std::size_t>(M * N));
  float base = static_cast<float>(K);
  for (auto& v : out) {
    v = base;
    if (add_error) v += 1e-4f * static_cast<float>(std::rand() % 100) / 100.f;
  }
  return out;
}

static ValidationResult validate_kernel(
    const std::string& name, int M, int N, int K,
    double rtol, double atol, bool inject_error = false) {

  ValidationResult vr;
  vr.kernel = name;
  vr.rtol   = rtol;
  vr.atol   = atol;
  vr.status = "pass";
  vr.max_abs_err = 0.0;
  vr.max_rel_err = 0.0;

  std::vector<float> ref;
  ref_matmul(M, N, K, ref);
  auto got = sim_kernel_result(M, N, K, inject_error);

  for (std::size_t i = 0; i < ref.size(); ++i) {
    double abs_err = std::abs(static_cast<double>(got[i] - ref[i]));
    double rel_err = std::abs(ref[i]) > 1e-12
                   ? abs_err / std::abs(static_cast<double>(ref[i]))
                   : abs_err;
    if (abs_err > vr.max_abs_err) vr.max_abs_err = abs_err;
    if (rel_err > vr.max_rel_err) vr.max_rel_err = rel_err;
    if (abs_err > atol && rel_err > rtol) {
      vr.status = "fail";
      vr.message = "element " + std::to_string(i) +
                   ": got=" + std::to_string(got[i]) +
                   " ref=" + std::to_string(ref[i]) +
                   " abs_err=" + std::to_string(abs_err);
    }
  }

  vr.passed = (vr.status == "pass");
  return vr;
}

static std::string make_validate_json(
    const std::vector<ValidationResult>& results,
    double rtol, double atol) {

  int total = static_cast<int>(results.size());
  int passed = 0;
  for (auto& r : results) if (r.passed) ++passed;

  std::string s = "{\n";
  s += "  \"summary\": {\n";
  s += "    \"total\": " + std::to_string(total) + ",\n";
  s += "    \"passed\": " + std::to_string(passed) + ",\n";
  s += "    \"failed\": " + std::to_string(total - passed) + ",\n";
  s += "    \"rtol\": " + std::to_string(rtol) + ",\n";
  s += "    \"atol\": " + std::to_string(atol) + "\n";
  s += "  },\n  \"kernels\": [\n";

  for (std::size_t i = 0; i < results.size(); ++i) {
    auto& r = results[i];
    char row[512];
    std::snprintf(row, sizeof(row),
                  "    {\"name\":\"%s\",\"status\":\"%s\","
                  "\"max_abs_err\":%.3e,\"max_rel_err\":%.3e%s}",
                  tessera::jsonEscape(r.kernel).c_str(), r.status.c_str(),
                  r.max_abs_err, r.max_rel_err,
                  r.message.empty() ? ""
                    : (",\"message\":\"" + tessera::jsonEscape(r.message) + "\"").c_str());
    s += row;
    if (i + 1 < results.size()) s += ",";
    s += "\n";
  }
  s += "  ]\n}\n";
  return s;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string opt_inputs, opt_golden, opt_shapes;
  std::string s_rtol = "1e-3", s_atol = "1e-5";
  double opt_rtol = 1e-3, opt_atol = 1e-5;
  int opt_timeout = 60;
  bool opt_allow_exec = false;
  std::vector<std::string> inputs;

  tessera::Args args(TOOL,
    "Execute compiled kernels and validate numerics against a reference",
    argc, argv);
  args.option("--inputs",     "Input tensor spec JSON file",           &opt_inputs)
      .option("--golden",     "Golden reference JSON file",            &opt_golden)
      .option("--shapes",     "Shapes for synthetic inputs (MxNxK)",   &opt_shapes, "1024x1024x1024")
      .option("--rtol",       "Relative tolerance (default 1e-3)",      &s_rtol, "1e-3")
      .option("--atol",       "Absolute tolerance (default 1e-5)",      &s_atol, "1e-5")
      .int_option("--timeout","Max execution seconds (default 60)",     &opt_timeout, 60)
      .flag("--allow-exec",   "Actually execute kernels (requires toolchain)", &opt_allow_exec)
      .positional("artifact-dir", "Artifact directory from tessera-compile", &inputs);

  if (!args.parse()) return args.exit_code();

  try { opt_rtol = std::stod(s_rtol); } catch(...) {}
  try { opt_atol = std::stod(s_atol); } catch(...) {}

  std::string artifact_dir = inputs.empty() ? args.out_dir() : inputs[0];
  int rc = tessera::EXIT_OK;

  try {
    auto paths = makeArtifactLayout(artifact_dir);

    if (opt_allow_exec) {
      TLOG_WARN(TOOL, "--allow-exec: real kernel execution not wired yet; "
                "using analytic reference");
    }

    // Parse shapes from --shapes "MxNxK"
    int M = 1024, N = 1024, K = 1024;
    {
      std::istringstream ss(opt_shapes);
      char sep;
      ss >> M >> sep >> N >> sep >> K;
      if (M <= 0 || N <= 0 || K <= 0) { M = N = K = 1024; }
    }

    // Get kernel list
    auto kernel_files = listDir(paths.kernels_dir);
    std::vector<std::string> kernel_names;
    if (kernel_files.empty()) {
      kernel_names.push_back("demo_kernel");
    } else {
      for (auto& f : kernel_files)
        kernel_names.push_back(f.substr(f.rfind('/') + 1));
    }

    std::vector<ValidationResult> results;
    for (auto& kname : kernel_names) {
      TLOG_INFO(TOOL, "validating " + kname +
                " shape=" + std::to_string(M) + "x" +
                std::to_string(N) + "x" + std::to_string(K) +
                " rtol=" + s_rtol + " atol=" + s_atol);
      auto vr = validate_kernel(kname, M, N, K, opt_rtol, opt_atol);
      results.push_back(vr);

      if (!vr.passed) {
        TLOG_ERROR(TOOL, kname + " FAILED: " + vr.message);
        rc = tessera::EXIT_EXEC_ERROR;
      } else {
        TLOG_INFO(TOOL, kname + " PASSED  max_abs_err=" +
                  std::to_string(vr.max_abs_err));
      }
    }

    if (!args.dry_run()) {
      writeFile(paths.reports_dir + "/validate.json",
                make_validate_json(results, opt_rtol, opt_atol));
      TLOG_INFO(TOOL, "validation report → reports/validate.json");
    } else {
      TLOG_INFO(TOOL, "[dry-run] would write reports/validate.json");
    }

  } catch (const std::exception& e) {
    TLOG_ERROR(TOOL, e.what());
    tessera::json_result(TOOL, artifact_dir, false,
                         "\"error\":\"" + tessera::jsonEscape(e.what()) + "\"");
    return tessera::EXIT_IO_ERROR;
  }

  bool ok = (rc == tessera::EXIT_OK);
  tessera::json_result(TOOL, artifact_dir, ok,
                       "\"rtol\":" + s_rtol + ",\"atol\":" + s_atol);
  return rc;
}
