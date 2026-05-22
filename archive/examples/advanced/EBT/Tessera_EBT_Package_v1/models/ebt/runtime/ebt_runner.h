#pragma once
#include <cstddef>
struct Tensor; // placeholder
struct EBTStepConfig { int T=4; int K=4; float eta=1e-1f; float noise=0.f; bool self_verify=true; };
void ebt_infer(const Tensor& x, const EBTStepConfig& cfg, Tensor* y_out);
