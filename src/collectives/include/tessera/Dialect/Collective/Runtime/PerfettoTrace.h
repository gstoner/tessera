#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <map>

namespace tessera { namespace collective {

struct TraceEvent {
  std::string name;
  std::string cat;
  char ph; // 'B','E','C'
  int pid=0;
  int tid=0;
  double ts_us=0.0;
  // optional args
  std::map<std::string, std::string> sargs;
  std::map<std::string, double> nargs;
};

class PerfettoTraceWriter {
public:
  void begin(const std::string& name, const std::string& cat, int pid, int tid) {
    TraceEvent e; e.name=name; e.cat=cat; e.ph='B'; e.pid=pid; e.tid=tid; e.ts_us=now();
    events_.push_back(e);
  }
  void end(const std::string& name, const std::string& cat, int pid, int tid) {
    TraceEvent e; e.name=name; e.cat=cat; e.ph='E'; e.pid=pid; e.tid=tid; e.ts_us=now();
    events_.push_back(e);
  }
  void counter(const std::string& name, double v, int pid, int tid) {
    TraceEvent e; e.name=name; e.cat="counter"; e.ph='C'; e.pid=pid; e.tid=tid; e.ts_us=now(); e.nargs["value"]=v;
    events_.push_back(e);
  }
  void annotate(const std::string& key, const std::string& val) {
    meta_[key] = val;
  }

  void write(const std::string& path) {
    std::ofstream f(path);
    f << "{\n";
    if (!meta_.empty()) {
      f << ""metadata": {";
      bool first=true;
      for (auto &kv : meta_) {
        if (!first) f << ",";
        f << "\"" << kv.first << "\": \"" << kv.second << "\"";
        first=false;
      }
      f << "},\n";
    }
    f << ""traceEvents": [\n";
    for (size_t i=0;i<events_.size();++i) {
      const auto &e=events_[i];
      f << " {";
      f << "\"name\":\"" << e.name << "\",";
      f << "\"cat\":\"" << e.cat << "\",";
      f << "\"ph\":\"" << e.ph << "\",";
      f << "\"pid\":" << e.pid << ",";
      f << "\"tid\":" << e.tid << ",";
      f << "\"ts\":" << e.ts_us;
      if (!e.sargs.empty() || !e.nargs.empty()) {
        f << ",\"args\":{";
        bool aFirst=true;
        for (auto &kv : e.sargs) {
          if (!aFirst) f << ","; aFirst=false;
          f << "\"" << kv.first << "\":\"" << kv.second << "\"";
        }
        for (auto &kv : e.nargs) {
          if (!aFirst) f << ","; aFirst=false;
          f << "\"" << kv.first << "\":" << kv.second;
        }
        f << "}";
      }
      f << " }";
      if (i+1<events_.size()) f << ",";
      f << "\n";
    }
    f << "] }\n";
  }

private:
  static double now() {
    using namespace std::chrono;
    static auto t0 = steady_clock::now();
    auto t = steady_clock::now();
    return duration<double, std::micro>(t - t0).count();
  }
  std::map<std::string,std::string> meta_;
  std::vector<TraceEvent> events_;
};

}} // ns
