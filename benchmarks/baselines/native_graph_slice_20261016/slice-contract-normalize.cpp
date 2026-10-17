#include "Tessera/IR/SliceContract.h"
#include <cstdint>
#include <iostream>
int main() {
  int64_t extent,start,size;
  while(std::cin>>extent>>start>>size) {
    if(extent<0) return 1;
    auto [offset,count]=tessera::slice_contract::normalizeAxis(extent,start,size);
    std::cout<<offset<<' '<<count<<'\n';
  }
}
