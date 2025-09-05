#include "harness/op_registry.h"

OpRegistry& OpRegistry::instance(){
  static OpRegistry reg;
  return reg;
}
void OpRegistry::add(const OpInfo& info){ map_[info.name]=info; }
std::vector<OpInfo> OpRegistry::list() const {
  std::vector<OpInfo> v;
  v.reserve(map_.size());
  for(auto& kv: map_) v.push_back(kv.second);
  return v;
}
OpFn OpRegistry::find(const std::string& name) const {
  auto it = map_.find(name);
  if(it==map_.end()) return nullptr;
  return it->second.fn;
}
