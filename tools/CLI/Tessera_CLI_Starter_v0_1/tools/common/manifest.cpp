#include "manifest.hpp"
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <ctime>

static void mkdir_p(const std::string& p) {
#ifdef _WIN32
  _mkdir(p.c_str());
#else
  ::mkdir(p.c_str(), 0755);
#endif
}

ArtifactPaths makeArtifactLayout(const std::string& out_dir) {
  ArtifactPaths a;
  a.out_dir = out_dir;
  a.ir_dir = out_dir + "/ir";
  a.kernels_dir = out_dir + "/kernels";
  a.host_dir = out_dir + "/host";
  a.cmake_dir = out_dir + "/cmake";
  a.reports_dir = out_dir + "/reports";
  a.tune_dir = out_dir + "/tune";
  a.meta_dir = out_dir + "/meta";
  mkdir_p(out_dir);
  mkdir_p(a.ir_dir);
  mkdir_p(a.kernels_dir);
  mkdir_p(a.host_dir);
  mkdir_p(a.cmake_dir);
  mkdir_p(a.reports_dir);
  mkdir_p(a.tune_dir);
  mkdir_p(a.meta_dir);
  return a;
}

std::string slurpFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

bool writeFile(const std::string& path, const std::string& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << data;
  return true;
}

std::string nowIso8601() {
  std::time_t t = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return std::string(buf);
}
