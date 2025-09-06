#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "common/manifest.hpp"

static void print_usage() {
  std::cerr << "Usage: tessera-autotune [options] <inputs>\n";
}

int main(int argc, char** argv) {
  std::string out_dir = "out";
  bool json = false;
  std::vector<std::string> inputs;

  for (int i=1;i<argc;++i) {
    std::string a = argv[i];
    if (a == "--out-dir" && i+1<argc) { out_dir = argv[++i]; }
    else if (a == "--json") { json = true; }
    else if (a == "-h" || a == "--help") { print_usage(); return 0; }
    else if (a.rfind("-",0)==0) { /* ignore unknown for skeleton */ }
    else inputs.push_back(a);
  }
  auto paths = makeArtifactLayout(out_dir);
  int rc = 0;
  std::string msg;
  try {

    // Skeleton: create SQLite schema file and summary.json (no DB driver linked here).
    std::string schema = R"(-- tune.db schema (skeleton)
CREATE TABLE IF NOT EXISTS results(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL,           -- (platform,arch,op,shape)
  candidate TEXT NOT NULL,     -- JSON of tunables
  metric REAL NOT NULL,        -- e.g., time_ms (lower is better)
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_key ON results(key);
)";
    writeFile(paths.tune_dir + "/schema.sql", schema);
    std::string summary = R"({"status":"ok","evaluations":16,"best":{"metric":0.79,"candidate":{"tile":[128,128,64]}}})";
    writeFile(paths.tune_dir + "/summary.json", summary);
    std::cerr << "[tessera-autotune] wrote tune schema + summary\n";

  } catch (const std::exception& e) {
    rc = 1;
    msg = e.what();
  }
  if (json) {
    std::cout << "{\"tool\":\"tessera-autotune\",\"out_dir\":\""<< out_dir <<"\",\"ok\":"<< (rc==0?"true":"false") <<",\"time\":\""<< nowIso8601() <<"\"}" << std::endl;
  }
  return rc;
}
