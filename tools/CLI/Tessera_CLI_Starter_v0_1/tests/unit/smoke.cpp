//===- smoke.cpp — Tessera CLI common-library unit tests -------------------===//
//
// Tests: makeArtifactLayout, slurpFile, writeFile, fileExists, fileSize,
//        listDir, jsonEscape, nowIso8601.
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>
#include "common/manifest.hpp"

// Simple test harness
static int g_tests = 0, g_failures = 0;
#define CHECK(cond) do { \
  ++g_tests; \
  if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s line %d\n", #cond, __LINE__); \
    ++g_failures; \
  } \
} while(0)
#define CHECK_THROW(expr) do { \
  ++g_tests; \
  bool threw = false; \
  try { (void)(expr); } catch(...) { threw = true; } \
  if (!threw) { \
    std::fprintf(stderr, "FAIL: expected exception from %s line %d\n", #expr, __LINE__); \
    ++g_failures; \
  } \
} while(0)

int main() {
  // ---- makeArtifactLayout -----------------------------------------------
  {
    auto p = makeArtifactLayout("smoke_test_out");
    CHECK(p.out_dir == "smoke_test_out");
    CHECK(p.ir_dir      == "smoke_test_out/ir");
    CHECK(p.kernels_dir == "smoke_test_out/kernels");
    CHECK(p.host_dir    == "smoke_test_out/host");
    CHECK(p.cmake_dir   == "smoke_test_out/cmake");
    CHECK(p.reports_dir == "smoke_test_out/reports");
    CHECK(p.tune_dir    == "smoke_test_out/tune");
    CHECK(p.meta_dir    == "smoke_test_out/meta");

    // All directories must have been created
    CHECK(fileExists("smoke_test_out"));
    CHECK(fileExists("smoke_test_out/ir"));
    CHECK(fileExists("smoke_test_out/kernels"));
    CHECK(fileExists("smoke_test_out/meta"));
  }

  // ---- writeFile / slurpFile / fileExists / fileSize --------------------
  {
    std::string path = "smoke_test_out/meta/test.txt";
    CHECK(writeFile(path, "hello tessera"));
    CHECK(fileExists(path));
    CHECK(fileSize(path) == 13);

    std::string content = slurpFile(path);
    CHECK(content == "hello tessera");
  }

  // ---- writeFile — binary round-trip ------------------------------------
  {
    std::string data(256, '\0');
    for (int i = 0; i < 256; ++i) data[i] = static_cast<char>(i);
    std::string path = "smoke_test_out/meta/binary.bin";
    CHECK(writeFile(path, data));
    CHECK(fileSize(path) == 256);
    CHECK(slurpFile(path) == data);
  }

  // ---- slurpFile — throws on missing file -------------------------------
  {
    CHECK_THROW(slurpFile("smoke_test_out/does_not_exist_xyz.txt"));
  }

  // ---- fileExists — non-existent returns false --------------------------
  {
    CHECK(!fileExists("smoke_test_out/nope_not_here_xyz"));
  }

  // ---- fileSize — non-existent returns -1 -------------------------------
  {
    CHECK(fileSize("smoke_test_out/nope_xyz") == -1L);
  }

  // ---- listDir ----------------------------------------------------------
  {
    writeFile("smoke_test_out/ir/a.mlir", "// a");
    writeFile("smoke_test_out/ir/b.mlir", "// b");
    auto files = listDir("smoke_test_out/ir");
    CHECK(files.size() >= 2);
    bool found_a = false, found_b = false;
    for (auto& f : files) {
      if (f.find("a.mlir") != std::string::npos) found_a = true;
      if (f.find("b.mlir") != std::string::npos) found_b = true;
    }
    CHECK(found_a);
    CHECK(found_b);
  }

  // ---- listDir — empty dir returns empty vector -------------------------
  {
    makeArtifactLayout("smoke_test_out/empty_dir_test");
    auto files = listDir("smoke_test_out/empty_dir_test/tune");
    CHECK(files.empty());
  }

  // ---- jsonEscape -------------------------------------------------------
  {
    CHECK(jsonEscape("hello") == "hello");
    CHECK(jsonEscape("say \"hi\"") == "say \\\"hi\\\"");
    CHECK(jsonEscape("a\\b") == "a\\\\b");
    CHECK(jsonEscape("line1\nline2") == "line1\\nline2");
    CHECK(jsonEscape("tab\there") == "tab\\there");
    CHECK(jsonEscape("cr\rhere") == "cr\\rhere");
    // Empty string
    CHECK(jsonEscape("") == "");
    // Path with forward slash (no escaping needed)
    CHECK(jsonEscape("/tmp/out") == "/tmp/out");
  }

  // ---- nowIso8601 -------------------------------------------------------
  {
    std::string ts = nowIso8601();
    // Must be "YYYY-MM-DDTHH:MM:SSZ" = 20 chars
    CHECK(ts.size() == 20);
    CHECK(ts[4] == '-');
    CHECK(ts[7] == '-');
    CHECK(ts[10] == 'T');
    CHECK(ts[13] == ':');
    CHECK(ts[16] == ':');
    CHECK(ts[19] == 'Z');
  }

  // ---- nested mkdir_p (via makeArtifactLayout with deep path) -----------
  {
    auto p = makeArtifactLayout("smoke_test_out/nested/deep/path");
    CHECK(fileExists("smoke_test_out/nested/deep/path"));
    CHECK(fileExists("smoke_test_out/nested/deep/path/ir"));
  }

  // ---- Cleanup (best-effort) --------------------------------------------
  // Not cleaning up so test output is inspectable; CI runs in temp dirs.

  std::fprintf(stdout, "Tessera CLI smoke: %d/%d passed\n",
               g_tests - g_failures, g_tests);
  return g_failures == 0 ? 0 : 1;
}
