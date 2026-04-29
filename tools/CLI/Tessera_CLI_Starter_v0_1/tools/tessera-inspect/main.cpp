//===- tessera-inspect/main.cpp — Summarize IR and compiled kernels --------===//
//
// tessera-inspect reads a compiled artifact directory (or a single .mlir /
// .ptx / .ll file) and emits a per-kernel summary table covering register
// usage, shared memory, estimated occupancy, code size, and entry names.
//
// Example usage:
//   tessera-inspect build/                  # full artifact dir
//   tessera-inspect build/ --format=csv     # machine-readable CSV
//   tessera-inspect model.mlir --show-ir    # dump IR ops summary
//   tessera-inspect build/ --kernel=attn --json
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "common/manifest.hpp"
#include "common/args.hpp"

static const char* TOOL = "tessera-inspect";

// ---------------------------------------------------------------------------
// Kernel descriptor
// ---------------------------------------------------------------------------

struct KernelDesc {
  std::string name;
  std::string file;
  int         regs          = 0;    // registers per thread
  int         smem_kb       = 0;    // static shared memory (KB)
  int         max_blocks_sm = 0;    // max concurrent blocks per SM
  double      occupancy     = 0.0;  // theoretical occupancy [0,1]
  long        size_bytes    = 0;    // binary size (bytes)
  std::string entry;                // mangled entry name
};

// Derive a plausible occupancy estimate from regs + smem
static double estimate_occupancy(int regs, int smem_kb, const std::string& arch) {
  // Very simplified model for sm_90/sm_80:
  // max 65536 regs per SM, 100KB smem per SM
  int sm_regs  = 65536;
  int sm_smem  = 102400;  // bytes
  int blk_size = 256;     // assume 256 threads/block

  if (regs <= 0) regs = 64;
  if (smem_kb <= 0) smem_kb = 0;

  int blocks_by_regs  = sm_regs  / std::max(1, regs * blk_size);
  int blocks_by_smem  = (smem_kb == 0)
                       ? 32
                       : sm_smem / std::max(1, smem_kb * 1024);
  int max_blocks      = std::min(blocks_by_regs, blocks_by_smem);
  int sm_max          = (arch.find("90") != std::string::npos) ? 32 : 16;
  max_blocks = std::min(max_blocks, sm_max);
  return std::min(1.0, static_cast<double>(max_blocks) / static_cast<double>(sm_max));
}

// Parse a PTX file for entry names, reg counts, and smem usage
static KernelDesc parse_ptx(const std::string& path) {
  KernelDesc kd;
  kd.file = path;
  kd.size_bytes = fileSize(path);

  std::string content;
  try { content = slurpFile(path); } catch(...) { return kd; }

  // Entry name: look for ".visible .entry <name>("
  std::string marker = ".visible .entry ";
  auto pos = content.find(marker);
  if (pos != std::string::npos) {
    pos += marker.size();
    auto paren = content.find('(', pos);
    if (paren != std::string::npos)
      kd.entry = content.substr(pos, paren - pos);
  }
  if (kd.entry.empty()) kd.entry = "(unknown)";
  kd.name = kd.entry;

  // Registers: look for ".reg .<type> %<name><N>;"  — count max reg index
  // Simplified: count ".reg" occurrences as proxy
  int reg_count = 0;
  std::size_t rpos = 0;
  while ((rpos = content.find(".reg ", rpos)) != std::string::npos) {
    ++reg_count; rpos += 5;
  }
  kd.regs = std::min(255, std::max(32, reg_count * 8));

  // Shared memory: look for ".shared" declarations
  int smem_count = 0;
  std::size_t spos = 0;
  while ((spos = content.find(".shared", spos)) != std::string::npos) {
    ++smem_count; spos += 7;
  }
  kd.smem_kb = std::min(100, smem_count * 4);

  // Target arch from .target directive
  std::string arch = "sm_90";
  auto tpos = content.find(".target ");
  if (tpos != std::string::npos) {
    tpos += 8;
    auto nl = content.find('\n', tpos);
    if (nl != std::string::npos) arch = content.substr(tpos, nl - tpos);
  }

  kd.occupancy = estimate_occupancy(kd.regs, kd.smem_kb, arch);
  kd.max_blocks_sm = std::max(1, static_cast<int>(kd.occupancy * 32));
  return kd;
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

static std::string format_markdown(const std::vector<KernelDesc>& kernels) {
  std::string s;
  s += "| kernel | regs | smem_kb | max_blks_sm | occupancy | size_bytes | entry |\n";
  s += "|--------|-----:|--------:|------------:|----------:|-----------:|-------|\n";
  for (auto& k : kernels) {
    char row[512];
    std::snprintf(row, sizeof(row),
                  "| %s | %d | %d | %d | %.2f | %ld | %s |\n",
                  k.name.c_str(), k.regs, k.smem_kb, k.max_blocks_sm,
                  k.occupancy, k.size_bytes, k.entry.c_str());
    s += row;
  }
  return s;
}

static std::string format_csv(const std::vector<KernelDesc>& kernels) {
  std::string s = "kernel,regs,smem_kb,max_blks_sm,occupancy,size_bytes,entry,file\n";
  for (auto& k : kernels) {
    char row[512];
    std::snprintf(row, sizeof(row),
                  "%s,%d,%d,%d,%.4f,%ld,%s,%s\n",
                  tessera::jsonEscape(k.name).c_str(),
                  k.regs, k.smem_kb, k.max_blocks_sm,
                  k.occupancy, k.size_bytes,
                  tessera::jsonEscape(k.entry).c_str(),
                  tessera::jsonEscape(k.file).c_str());
    s += row;
  }
  return s;
}

static std::string format_json(const std::vector<KernelDesc>& kernels) {
  std::string s = "{\"kernels\":[\n";
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    auto& k = kernels[i];
    char row[512];
    std::snprintf(row, sizeof(row),
                  "  {\"name\":\"%s\",\"regs\":%d,\"smem_kb\":%d,"
                  "\"max_blocks_sm\":%d,\"occupancy\":%.4f,"
                  "\"size_bytes\":%ld,\"entry\":\"%s\",\"file\":\"%s\"}",
                  tessera::jsonEscape(k.name).c_str(),
                  k.regs, k.smem_kb, k.max_blocks_sm,
                  k.occupancy, k.size_bytes,
                  tessera::jsonEscape(k.entry).c_str(),
                  tessera::jsonEscape(k.file).c_str());
    s += row;
    if (i + 1 < kernels.size()) s += ",";
    s += "\n";
  }
  s += "]}\n";
  return s;
}

static void format_table(const std::vector<KernelDesc>& kernels) {
  // Aligned text table to stdout
  const char* HDR = "%-30s %6s %8s %12s %9s %11s\n";
  const char* ROW = "%-30s %6d %8d %12d %8.1f%% %11ld\n";
  std::printf(HDR, "KERNEL", "REGS", "SMEM_KB", "MAX_BLKS_SM", "OCCUPANCY", "SIZE_BYTES");
  std::printf("%s\n", std::string(80, '-').c_str());
  for (auto& k : kernels) {
    std::printf(ROW, k.name.c_str(), k.regs, k.smem_kb,
                k.max_blocks_sm, k.occupancy * 100.0, k.size_bytes);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  std::string opt_format, opt_kernel;
  bool opt_show_ir  = false;
  bool opt_show_ptx = false;
  std::vector<std::string> inputs;

  tessera::Args args(TOOL,
    "Summarize compiled kernels (regs, smem, occupancy, size, entry)",
    argc, argv);
  args.option("--format",  "Output format {table,md,csv,json} (default table)", &opt_format, "table")
      .option("--kernel",  "Filter: inspect only this kernel name",              &opt_kernel)
      .flag("--show-ir",   "Include IR op-count summary",                        &opt_show_ir)
      .flag("--show-ptx",  "Print full PTX content for each kernel",             &opt_show_ptx)
      .positional("path",  "Artifact dir or individual .ptx/.mlir file",         &inputs);

  if (!args.parse()) return args.exit_code();

  // Validate format
  if (opt_format != "table" && opt_format != "md" &&
      opt_format != "csv"   && opt_format != "json") {
    TLOG_ERROR(TOOL, "unknown format '" + opt_format +
               "'; expected table, md, csv, or json");
    return tessera::EXIT_PARSE_ERROR;
  }

  std::string target = inputs.empty() ? args.out_dir() : inputs[0];
  int rc = tessera::EXIT_OK;

  try {
    std::vector<KernelDesc> kernels;

    // If target is a .ptx file directly
    if (target.size() > 4 && target.substr(target.size() - 4) == ".ptx") {
      if (!fileExists(target)) {
        TLOG_ERROR(TOOL, "file not found: " + target);
        return tessera::EXIT_IO_ERROR;
      }
      KernelDesc kd = parse_ptx(target);
      if (opt_kernel.empty() || kd.name.find(opt_kernel) != std::string::npos)
        kernels.push_back(kd);
    } else {
      // Treat as artifact directory
      auto paths = makeArtifactLayout(target);
      auto files = listDir(paths.kernels_dir);

      if (files.empty()) {
        // No compiled kernels — generate synthetic descriptor for demo
        TLOG_WARN(TOOL, "no kernel files in " + paths.kernels_dir +
                  "; showing synthetic descriptor");
        KernelDesc kd;
        kd.name         = "demo_kernel";
        kd.file         = "(not compiled)";
        kd.regs         = 128;
        kd.smem_kb      = 48;
        kd.occupancy    = estimate_occupancy(128, 48, "sm_90");
        kd.max_blocks_sm= static_cast<int>(kd.occupancy * 32);
        kd.size_bytes   = 0;
        kd.entry        = "demo_kernel";
        kernels.push_back(kd);
      } else {
        for (auto& f : files) {
          // Only parse PTX for now; skip other formats
          if (f.size() < 4 || f.substr(f.size() - 4) != ".ptx") {
            KernelDesc kd;
            kd.name = f.substr(f.rfind('/') + 1);
            kd.file = f;
            kd.size_bytes = fileSize(f);
            kd.entry = "(non-ptx)";
            if (opt_kernel.empty() || kd.name.find(opt_kernel) != std::string::npos)
              kernels.push_back(kd);
            continue;
          }
          KernelDesc kd = parse_ptx(f);
          if (opt_kernel.empty() || kd.name.find(opt_kernel) != std::string::npos)
            kernels.push_back(kd);
        }
      }

      // IR summary if requested
      if (opt_show_ir) {
        auto ir_files = listDir(paths.ir_dir);
        std::cout << "\n--- IR SNAPSHOTS ---\n";
        for (auto& f : ir_files) {
          std::cout << "  " << f << "  (" << fileSize(f) << " bytes)\n";
        }
        std::cout << "\n";
      }
    }

    if (kernels.empty()) {
      TLOG_ERROR(TOOL, "no matching kernels to inspect");
      return tessera::EXIT_PARSE_ERROR;
    }

    // Show PTX if requested
    if (opt_show_ptx) {
      for (auto& k : kernels) {
        if (k.file != "(not compiled)" && k.file != "(non-ptx)") {
          std::cout << "\n--- PTX: " << k.file << " ---\n";
          try { std::cout << slurpFile(k.file) << "\n"; } catch(...) {}
        }
      }
    }

    // Output
    std::string out_path = args.output();
    bool to_stdout = (out_path == "-");

    if (opt_format == "table") {
      format_table(kernels);
    } else {
      std::string content;
      if (opt_format == "md")       content = format_markdown(kernels);
      else if (opt_format == "csv") content = format_csv(kernels);
      else /* json */               content = format_json(kernels);

      // Also save to reports/
      auto paths = makeArtifactLayout(target);
      std::string report_ext = (opt_format == "md") ? ".md"
                             : (opt_format == "csv") ? ".csv" : ".json";
      if (!args.dry_run()) {
        writeFile(paths.reports_dir + "/inspect" + report_ext, content);
        TLOG_INFO(TOOL, "wrote reports/inspect" + report_ext);
      }

      if (to_stdout) std::cout << content;
      else if (!args.dry_run()) writeFile(out_path, content);
    }

  } catch (const std::exception& e) {
    TLOG_ERROR(TOOL, e.what());
    tessera::json_result(TOOL, target, false,
                         "\"error\":\"" + tessera::jsonEscape(e.what()) + "\"");
    return tessera::EXIT_IO_ERROR;
  }

  tessera::json_result(TOOL, target, rc == tessera::EXIT_OK,
                       "\"format\":\"" + opt_format + "\"");
  return rc;
}
