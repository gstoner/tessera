#include "TesseraROCM/MemSpace.h"
#include "mlir/IR/Attributes.h"
using namespace mlir;
using namespace mlir::tessera_rocm;
const char* MemSpace::kindToStr(Kind k){
  switch(k){case Global:return "global";case LDS:return "lds";case Private:return "private";default:return "unknown";}
}
unsigned MemSpace::toAddressSpace(Kind k){
  switch(k){case Global:return 1;case LDS:return 3;case Private:return 5;default:return 0;}
}
std::optional<MemSpace> MemSpace::parse(Attribute a){
  if (auto s = a.dyn_cast_or_null<StringAttr>()){
    MemSpace m; auto v=s.str();
    if(v=="global")m.kind=Global; else if(v=="lds")m.kind=LDS; else if(v=="private")m.kind=Private; else m.kind=Unknown;
    return m;
  }
  return std::nullopt;
}
