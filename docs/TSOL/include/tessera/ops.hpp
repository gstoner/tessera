
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace tessera {
enum class DType:int32_t{FP32,BF16,FP16,FP8_E4M3,FP8_E5M2,INT8,INT32};
enum class Layout:int32_t{ROW_MAJOR,COL_MAJOR,NHWC,NCHW,TILE,BSR};
struct Tensor{void*data;DType dtype;Layout layout;int64_t*shape;int32_t rank;void*meta;};
struct Epilogue{bool add_bias=false;Tensor bias{};enum Act{NONE,RELU,SILU,GELU} activation=NONE;bool add_residual=false;Tensor residual{};};
struct Determinism{bool deterministic=false;};
struct FlashParams{bool causal=false;int block_q=128,block_k=128,block_d=128;};
void matmul(const Tensor&,const Tensor&,Tensor&,const Epilogue* =nullptr) noexcept(false);
void conv2d(const Tensor&,const Tensor&,Tensor&,int,int,int,int,const Epilogue* =nullptr) noexcept(false);
void layernorm(const Tensor&,Tensor&,float) noexcept(false);
void qkv_projection(const Tensor&,const Tensor&,Tensor&,Tensor&,Tensor&) noexcept(false);
void flash_attention(const Tensor&,const Tensor&,const Tensor&,Tensor&,const FlashParams& ={}, const Determinism* =nullptr) noexcept(false);
} // namespace tessera
