#pragma once
#include <cstdint>
namespace tessera { namespace ebt {
struct EBTPipelineOptions { int32_t K=4; int32_t T=4; bool useJVP=false; };
void registerEBTPipelines();     // call once in main()
}} // ns
