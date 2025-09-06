#include <cassert>
#include <string>
#include "tools/common/manifest.hpp"

int main() {
  auto paths = makeArtifactLayout("out_test");
  assert(paths.out_dir.find("out_test") != std::string::npos);
  // Write & read
  std::string p = paths.meta_dir + "/hello.txt";
  assert(writeFile(p, "ok"));
  auto s = slurpFile(p);
  assert(s == "ok");
  return 0;
}
