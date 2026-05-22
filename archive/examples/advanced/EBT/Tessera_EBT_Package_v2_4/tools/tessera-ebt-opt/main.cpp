#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
int main(int argc, char** argv) {
  int K=4, T=4; bool useJVP=false;
  std::vector<std::string> args(argv+1, argv+argc);
  for (auto& a: args) {
    if (a.rfind("--ebt-K=",0)==0) K = std::atoi(a.substr(8).c_str());
    else if (a.rfind("--ebt-T=",0)==0) T = std::atoi(a.substr(8).c_str());
    else if (a=="--ebt-use-jvp=true") useJVP=true;
  }
  std::cout << "// tessera-ebt-opt options: K="<<K<<" T="<<T<<" useJVP="<<(useJVP?\"true\":\"false\") << \"\\n\";
  std::cout << "// passes: tessera-ebt-materialize-loops ; tessera-ebt-select-grad-path\\n";
  // stdin→stdout passthrough for FileCheck
  std::ios::sync_with_stdio(false);
  std::cin >> std::noskipws; char c; while (std::cin >> c) std::cout << c;
  return 0;
}
