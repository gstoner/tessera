#pragma once
#include <string>
#include <vector>
#include <map>

struct ArtifactPaths {
  std::string out_dir;
  std::string ir_dir;
  std::string kernels_dir;
  std::string host_dir;
  std::string cmake_dir;
  std::string reports_dir;
  std::string tune_dir;
  std::string meta_dir;
};

ArtifactPaths makeArtifactLayout(const std::string& out_dir);

std::string slurpFile(const std::string& path);
bool writeFile(const std::string& path, const std::string& data);
std::string nowIso8601();
