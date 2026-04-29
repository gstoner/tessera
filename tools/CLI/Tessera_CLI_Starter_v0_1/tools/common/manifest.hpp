//===- manifest.hpp — Artifact layout and filesystem helpers ---------------===//
#pragma once
#include <string>
#include <vector>

// Version kept in sync with args.hpp TESSERA_CLI_VERSION
#define TESSERA_CLI_VERSION "0.4.0"

namespace tessera {

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

/// Create the full artifact directory tree under out_dir.
/// All intermediate directories are created (recursive mkdir).
ArtifactPaths makeArtifactLayout(const std::string& out_dir);

/// Ensure a single directory path exists (recursive mkdir_p).
void ensureDir(const std::string& path);

/// Read an entire file into a string.
/// Throws std::runtime_error if the file cannot be opened.
std::string slurpFile(const std::string& path);

/// Write data to path.  Returns true on success; throws on I/O error.
bool writeFile(const std::string& path, const std::string& data);

/// Return true if path exists and is readable.
bool fileExists(const std::string& path);

/// Return file size in bytes, or -1 if not found.
long fileSize(const std::string& path);

/// List files directly inside dir (non-recursive).
std::vector<std::string> listDir(const std::string& dir);

/// Current UTC time in ISO-8601 format (e.g. "2026-04-29T14:32:01Z").
std::string nowIso8601();

/// Escape a string for embedding inside a JSON value (quotes not included).
std::string jsonEscape(const std::string& s);

} // namespace tessera

// ---------------------------------------------------------------------------
// Backward-compat shims — allow unqualified calls from legacy code
// ---------------------------------------------------------------------------
using tessera::ArtifactPaths;
using tessera::makeArtifactLayout;
using tessera::ensureDir;
using tessera::slurpFile;
using tessera::writeFile;
using tessera::fileExists;
using tessera::fileSize;
using tessera::listDir;
using tessera::nowIso8601;
using tessera::jsonEscape;
