
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
  int K = 4, T = 4;
  bool useJVP = false;
  std::string reportDir;
  std::vector<std::string> args(argv+1, argv+argc);
  for (auto &a : args) {
    if (a.rfind("--ebt-K=",0)==0) K = std::atoi(a.substr(8).c_str());
    else if (a.rfind("--ebt-T=",0)==0) T = std::atoi(a.substr(8).c_str());
    else if (a=="--ebt-use-jvp=true") useJVP = true;
    else if (a.rfind("--report=",0)==0) reportDir = a.substr(9);
  }
  std::cout << "// tessera-ebt-opt options: K="<<K<<" T="<<T<<" useJVP="<<(useJVP?"true":"false")<<"\n";
  std::cout << "// pipeline: tessera-ebt-canonicalize | tessera-ebt-lower\n";

  if (!reportDir.empty()) {
    // Write a tiny roofline.html and perfetto.json to the chosen dir (scaffold demo).
    std::ofstream h(reportDir + "/roofline.html");
    h << "<!doctype html><meta charset='utf-8'><title>EBT Roofline</title>\n";
    h << "<h1>EBT Roofline (demo)</h1><p>K="<<K<<" T="<<T<<" useJVP="<<(useJVP?"true":"false")<<"</p>\n";
    h << "<p>This is a placeholder. Hook your real CSV → chart code here.</p>\n";
    h.close();
    std::ofstream p(reportDir + "/perfetto.json");
    p << "{\"traceEvents\":[{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"args\":{\"name\":\"tessera-ebt\"}},"
         "{\"ph\":\"X\",\"name\":\"EBT inner loop\",\"ts\":0,\"dur\":1000,\"pid\":1,\"tid\":0,\"args\":{\"K\":"<<K<<",\"T\":"<<T<<"}}]}";
    p.close();
    std::cout << "// report: wrote roofline.html and perfetto.json\n";
  }

  // Echo stdin to stdout (so FileCheck can be used with RUN lines in samples).
  std::ios::sync_with_stdio(false);
  std::cin >> std::noskipws;
  char c;
  while (std::cin >> c) std::cout << c;
  return 0;
}
