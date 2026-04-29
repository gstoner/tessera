//===- manifest.cpp — Artifact layout and filesystem helpers ---------------===//
#include "manifest.hpp"
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ctime>
#include <cstring>
#include <cerrno>

namespace tessera {

// ---------------------------------------------------------------------------
// mkdir_p — internal helper; create a directory and all missing parents (POSIX)
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& path) {
  std::string tmp = path;
  // Ensure trailing slash so we can loop through all components
  if (tmp.empty()) return;
  if (tmp.back() != '/') tmp += '/';

  for (std::size_t pos = 1; pos < tmp.size(); ++pos) {
    if (tmp[pos] == '/') {
      tmp[pos] = '\0';
#ifdef _WIN32
      _mkdir(tmp.c_str());
#else
      ::mkdir(tmp.c_str(), 0755);
#endif
      tmp[pos] = '/';
    }
  }
}

// ---------------------------------------------------------------------------
// ensureDir — public wrapper around mkdir_p
// ---------------------------------------------------------------------------
void ensureDir(const std::string& path) {
  mkdir_p(path);
}

// ---------------------------------------------------------------------------
// makeArtifactLayout
// ---------------------------------------------------------------------------
ArtifactPaths makeArtifactLayout(const std::string& out_dir) {
  ArtifactPaths a;
  a.out_dir      = out_dir;
  a.ir_dir       = out_dir + "/ir";
  a.kernels_dir  = out_dir + "/kernels";
  a.host_dir     = out_dir + "/host";
  a.cmake_dir    = out_dir + "/cmake";
  a.reports_dir  = out_dir + "/reports";
  a.tune_dir     = out_dir + "/tune";
  a.meta_dir     = out_dir + "/meta";

  mkdir_p(a.out_dir);
  mkdir_p(a.ir_dir);
  mkdir_p(a.kernels_dir);
  mkdir_p(a.host_dir);
  mkdir_p(a.cmake_dir);
  mkdir_p(a.reports_dir);
  mkdir_p(a.tune_dir);
  mkdir_p(a.meta_dir);
  return a;
}

// ---------------------------------------------------------------------------
// slurpFile — throws if file not found
// ---------------------------------------------------------------------------
std::string slurpFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("cannot open file '" + path + "': " +
                             std::strerror(errno));
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

// ---------------------------------------------------------------------------
// writeFile — throws on I/O error
// ---------------------------------------------------------------------------
bool writeFile(const std::string& path, const std::string& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("cannot write file '" + path + "': " +
                             std::strerror(errno));
  }
  ofs << data;
  if (!ofs.good()) {
    throw std::runtime_error("I/O error writing '" + path + "'");
  }
  return true;
}

// ---------------------------------------------------------------------------
// fileExists
// ---------------------------------------------------------------------------
bool fileExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

// ---------------------------------------------------------------------------
// fileSize
// ---------------------------------------------------------------------------
long fileSize(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return -1L;
  return static_cast<long>(st.st_size);
}

// ---------------------------------------------------------------------------
// listDir
// ---------------------------------------------------------------------------
std::vector<std::string> listDir(const std::string& dir) {
  std::vector<std::string> result;
#ifdef _WIN32
  // Windows: use FindFirstFile / FindNextFile (not implemented here)
  (void)dir;
#else
  DIR* d = ::opendir(dir.c_str());
  if (!d) return result;
  struct dirent* ent;
  while ((ent = ::readdir(d)) != nullptr) {
    std::string name = ent->d_name;
    if (name == "." || name == "..") continue;
    result.push_back(dir + "/" + name);
  }
  ::closedir(d);
#endif
  return result;
}

// ---------------------------------------------------------------------------
// nowIso8601
// ---------------------------------------------------------------------------
std::string nowIso8601() {
  std::time_t t = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return std::string(buf);
}

// ---------------------------------------------------------------------------
// jsonEscape
// ---------------------------------------------------------------------------
std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

} // namespace tessera
