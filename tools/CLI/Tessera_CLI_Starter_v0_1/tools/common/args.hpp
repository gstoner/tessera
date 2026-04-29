//===- args.hpp — Shared CLI argument parser for Tessera tools ------------===//
//
// Provides a thin argument-parsing layer used by every tessera-* tool so that
// flag handling is defined once and flags stay consistent across the suite.
//
// Usage:
//   #include "common/args.hpp"
//
//   int main(int argc, char** argv) {
//     tessera::Args args("tessera-opt", "Apply Tessera/MLIR passes", argc, argv);
//     args.flag("--verify",     "Run verifiers after each phase", &opt_verify);
//     args.option("--pipeline", "Pass-pipeline string",           &opt_pipeline);
//     args.positional("input",  "Input .mlir file(s)",            &inputs);
//     if (!args.parse()) return args.exit_code();
//     // ... use parsed values ...
//   }
//===----------------------------------------------------------------------===//

#pragma once
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define TESSERA_CLI_VERSION "0.4.0"

namespace tessera {

// ---------------------------------------------------------------------------
// Exit codes (matches CLI_Design.md §Exit Codes)
// ---------------------------------------------------------------------------
enum ExitCode : int {
  EXIT_OK            = 0,
  EXIT_PARSE_ERROR   = 1,   // bad flags / IR parse error
  EXIT_PASS_ERROR    = 2,   // pass pipeline failure
  EXIT_IO_ERROR      = 3,   // file read/write/perm
  EXIT_LOWER_ERROR   = 10,  // lowering failure
  EXIT_TOOLCHAIN     = 11,  // backend toolchain failure
  EXIT_BAD_TARGET    = 12,  // invalid target/arch
  EXIT_EXEC_ERROR    = 20,  // runtime execution failure
  EXIT_TUNE_ERROR    = 30,  // autotune failure
};

// ---------------------------------------------------------------------------
// Log levels
// ---------------------------------------------------------------------------
enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERROR };

static LogLevel g_log_level = LogLevel::INFO;
static bool     g_use_json  = false;

inline bool log_enabled(LogLevel l) { return l >= g_log_level; }

inline void log(LogLevel l, const std::string& tool, const std::string& msg) {
  if (!log_enabled(l)) return;
  const char* prefix = "";
  switch (l) {
    case LogLevel::TRACE: prefix = "TRACE"; break;
    case LogLevel::DEBUG: prefix = "DEBUG"; break;
    case LogLevel::INFO:  prefix = "INFO";  break;
    case LogLevel::WARN:  prefix = "WARN";  break;
    case LogLevel::ERROR: prefix = "ERROR"; break;
  }
  std::cerr << "[" << tool << "] " << prefix << ": " << msg << "\n";
}

#define TLOG_INFO(tool, msg)  tessera::log(tessera::LogLevel::INFO,  tool, msg)
#define TLOG_WARN(tool, msg)  tessera::log(tessera::LogLevel::WARN,  tool, msg)
#define TLOG_ERROR(tool, msg) tessera::log(tessera::LogLevel::ERROR, tool, msg)
#define TLOG_DEBUG(tool, msg) tessera::log(tessera::LogLevel::DEBUG, tool, msg)

// ---------------------------------------------------------------------------
// Args — argument parser
// ---------------------------------------------------------------------------

class Args {
public:
  Args(const std::string& tool_name,
       const std::string& description,
       int argc, char** argv)
    : tool_(tool_name), desc_(description), argc_(argc), argv_(argv) {}

  // Register a boolean flag (--flag sets *dest = true)
  Args& flag(const std::string& name, const std::string& help, bool* dest) {
    flags_[name] = {help, dest};
    return *this;
  }

  // Register a string option (--opt <value>)
  Args& option(const std::string& name, const std::string& help, std::string* dest,
               const std::string& default_val = "") {
    *dest = default_val;
    opts_[name] = {help, dest};
    return *this;
  }

  // Register an int option (--opt <int>)
  Args& int_option(const std::string& name, const std::string& help, int* dest,
                   int default_val = 0) {
    *dest = default_val;
    int_opts_[name] = {help, dest};
    return *this;
  }

  // Register a repeatable string option (--flag value can appear N times)
  Args& multi(const std::string& name, const std::string& help,
              std::vector<std::string>* dest) {
    multi_[name] = {help, dest};
    return *this;
  }

  // Register a positional accumulator (all non-flag args go here)
  Args& positional(const std::string& metavar, const std::string& help,
                   std::vector<std::string>* dest) {
    pos_dest_ = dest;
    pos_help_ = help;
    pos_meta_ = metavar;
    return *this;
  }

  // ---------------------------------------------------------------------------
  // parse() — process argv[1..argc-1].
  // Returns true on success, false on error (call exit_code() for the code).
  // ---------------------------------------------------------------------------
  bool parse() {
    // Wire built-in flags
    bool help = false, version = false, verbose = false;
    flag("--help",    "Show this help message and exit", &help);
    flag("-h",        "Show this help message and exit", &help);
    flag("--version", "Print tessera CLI version and exit", &version);
    flag("--json",    "Emit machine-readable JSON summary to stdout", &json_);
    flag("--verbose", "Increase log verbosity to INFO", &verbose);
    flag("--dry-run", "Print what would happen without doing it", &dry_run_);
    flag("--fail-on-warn", "Promote warnings to errors", &fail_on_warn_);

    option("--out-dir",   "Root for multi-file artifacts", &out_dir_, "out");
    option("-o",          "Output file (use - for stdout)",  &output_,  "-");
    option("--log-level", "Log verbosity {error,warn,info,debug,trace}",
           &log_level_str_, "info");
    option("--temp-dir",  "Temporary workspace", &temp_dir_, "/tmp/tessera");
    option("--cache-dir", "Pass/codegen cache location", &cache_dir_, "");
    multi("-I",           "Add import path", &include_dirs_);

    for (int i = 1; i < argc_; ++i) {
      std::string a = argv_[i];

      // Boolean flags
      if (auto it = flags_.find(a); it != flags_.end()) {
        *it->second.dest = true;
        continue;
      }

      // String options (--opt value or --opt=value)
      bool matched_opt = false;
      for (auto& [name, od] : opts_) {
        if (a == name) {
          if (i + 1 >= argc_) {
            std::cerr << "error: " << name << " requires a value\n";
            ec_ = EXIT_PARSE_ERROR; return false;
          }
          *od.dest = argv_[++i];
          matched_opt = true; break;
        }
        if (a.rfind(name + "=", 0) == 0) {
          *od.dest = a.substr(name.size() + 1);
          matched_opt = true; break;
        }
      }
      if (matched_opt) continue;

      // Int options (--opt value  OR  --opt=value)
      bool matched_int = false;
      for (auto& [name, id] : int_opts_) {
        std::string val_str;
        if (a == name) {
          if (i + 1 >= argc_) {
            std::cerr << "error: " << name << " requires an integer value\n";
            ec_ = EXIT_PARSE_ERROR; return false;
          }
          val_str = argv_[++i];
        } else if (a.rfind(name + "=", 0) == 0) {
          val_str = a.substr(name.size() + 1);
        } else {
          continue;
        }
        try { *id.dest = std::stoi(val_str); }
        catch (...) {
          std::cerr << "error: " << name << " expects an integer\n";
          ec_ = EXIT_PARSE_ERROR; return false;
        }
        matched_int = true; break;
      }
      if (matched_int) continue;

      // Multi options
      bool matched_multi = false;
      for (auto& [name, md] : multi_) {
        if (a == name) {
          if (i + 1 >= argc_) {
            std::cerr << "error: " << name << " requires a value\n";
            ec_ = EXIT_PARSE_ERROR; return false;
          }
          md.dest->push_back(argv_[++i]);
          matched_multi = true; break;
        }
        if (a.rfind(name + "=", 0) == 0) {
          md.dest->push_back(a.substr(name.size() + 1));
          matched_multi = true; break;
        }
      }
      if (matched_multi) continue;

      // Unknown flag
      if (a.rfind("-", 0) == 0) {
        std::cerr << "error: unknown option '" << a << "'\n"
                  << "Run with --help for usage.\n";
        ec_ = EXIT_PARSE_ERROR; return false;
      }

      // Positional
      if (pos_dest_) pos_dest_->push_back(a);
    }

    // Handle built-ins after parsing so --help works even with bad flags
    if (help)    { print_help(); ec_ = EXIT_OK; return false; }
    if (version) { print_version(); ec_ = EXIT_OK; return false; }

    // Apply log level
    if (verbose) log_level_str_ = "debug";
    apply_log_level();

    g_use_json = json_;
    return true;
  }

  // Accessors
  int         exit_code()    const { return ec_; }
  bool        json()         const { return json_; }
  bool        dry_run()      const { return dry_run_; }
  bool        fail_on_warn() const { return fail_on_warn_; }
  std::string out_dir()      const { return out_dir_; }
  std::string output()       const { return output_; }
  std::string temp_dir()     const { return temp_dir_; }
  std::string cache_dir()    const { return cache_dir_; }
  const std::vector<std::string>& include_dirs() const { return include_dirs_; }

private:
  struct FlagDef  { std::string help; bool* dest; };
  struct OptDef   { std::string help; std::string* dest; };
  struct IntOptDef{ std::string help; int* dest; };
  struct MultiDef { std::string help; std::vector<std::string>* dest; };

  std::string tool_, desc_;
  int argc_; char** argv_;
  int  ec_          = EXIT_OK;
  bool json_        = false;
  bool dry_run_     = false;
  bool fail_on_warn_= false;
  std::string out_dir_, output_, log_level_str_, temp_dir_, cache_dir_;
  std::vector<std::string> include_dirs_;
  std::vector<std::string>* pos_dest_ = nullptr;
  std::string pos_help_, pos_meta_;

  std::unordered_map<std::string, FlagDef>   flags_;
  std::unordered_map<std::string, OptDef>    opts_;
  std::unordered_map<std::string, IntOptDef> int_opts_;
  std::unordered_map<std::string, MultiDef>  multi_;

  void apply_log_level() {
    const std::string& l = log_level_str_;
    if (l == "trace") g_log_level = LogLevel::TRACE;
    else if (l == "debug") g_log_level = LogLevel::DEBUG;
    else if (l == "info")  g_log_level = LogLevel::INFO;
    else if (l == "warn")  g_log_level = LogLevel::WARN;
    else if (l == "error") g_log_level = LogLevel::ERROR;
    else {
      std::cerr << "warning: unknown --log-level '" << l
                << "'; using 'info'\n";
    }
  }

  void print_version() const {
    std::cout << tool_ << " " << TESSERA_CLI_VERSION << "\n";
  }

  void print_help() const {
    std::cout << "USAGE: " << tool_ << " [options] ";
    if (pos_dest_) std::cout << "<" << pos_meta_ << ">...";
    std::cout << "\n\n" << desc_ << "\n\nOPTIONS:\n";

    auto print_row = [](const std::string& n, const std::string& h) {
      std::cout << "  " << n;
      if (n.size() < 26)
        std::cout << std::string(26 - n.size(), ' ');
      else
        std::cout << "\n" << std::string(28, ' ');
      std::cout << h << "\n";
    };

    for (auto& [n, f] : flags_)    print_row(n, f.help);
    for (auto& [n, o] : opts_)     print_row(n + " <val>", o.help);
    for (auto& [n, o] : int_opts_) print_row(n + " <int>", o.help);
    for (auto& [n, m] : multi_)    print_row(n + " <val>", m.help + " (repeatable)");
    if (pos_dest_) print_row("<" + pos_meta_ + ">", pos_help_);
    std::cout << "\n";
  }
};

// ---------------------------------------------------------------------------
// json_result() — emit the standard one-line JSON summary to stdout
// ---------------------------------------------------------------------------
inline void json_result(const std::string& tool, const std::string& out_dir,
                        bool ok, const std::string& extra = "") {
  if (!g_use_json) return;
  std::cout << "{\"tool\":\"" << tool << "\","
            << "\"version\":\"" << TESSERA_CLI_VERSION << "\","
            << "\"out_dir\":\"" << out_dir << "\","
            << "\"ok\":" << (ok ? "true" : "false") << ","
            << "\"time\":\"";
  {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    std::cout << buf;
  }
  std::cout << "\"";
  if (!extra.empty()) std::cout << "," << extra;
  std::cout << "}" << std::endl;
}

} // namespace tessera
