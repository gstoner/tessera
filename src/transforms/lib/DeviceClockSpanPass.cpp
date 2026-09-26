//===- DeviceClockSpanPass.cpp - kernel-side clock span instrumentation --===//
//
// Stamp every kernel's execution span with the target's constant-rate device
// clock, so a native package can carry its own kernel-side timing witness
// (sync WSL-TIMING-ADMISSION-2026-09-26; owner direction, MASTER_AUDIT
// 2026-09-25: missing /dev/kfd or bare metal must not block measuring the
// compiler).
//
// This is the compiler-owned form of the pattern the Python HIP emitter
// validated on gfx1151 (DEVICE-CLOCK-DISCIPLINE-2026-08-31), where the
// device clock agreed with HIP events and the host wall clock to four
// significant figures. The emitter is a source generator the alpha guard rails
// forbid as a lowering route, so the native MLIR route needs its own.
//
// For each `gpu.func ... kernel` the pass
//   * appends one `!llvm.ptr<1>` argument: a two-word i64 span buffer the host
//     initializes to {UINT64_MAX, 0};
//   * at entry: thread (0,0,0) of each block does atomic umin(span[0], clock),
//     then every thread meets a `gpu.barrier` so no wave starts work before
//     the start stamp;
//   * before the kernel's single final `gpu.return`: a `gpu.barrier` so every
//     wave has finished, then thread (0,0,0) does atomic umax(span[1], clock).
// Reduced across blocks (and across launches that reuse the buffer), the span
// is the whole kernel, not one workgroup's slice.
//
// The barriers are load-bearing: without the end barrier, wave 0 of a
// multi-wave block stamps the end while another wave still computes -- an
// UNDER-estimate, which makes a kernel look fast and gets published. A barrier
// is only safe where every thread reaches it, so the pass REFUSES any kernel
// whose `gpu.return` is not the single terminator of the body's only block
// (an early return in divergent control flow would deadlock at the barrier).
//
// Clocks, per backend:
//   rocm   -- llvm.readsteadycounter: on gfx11/gfx12 this lowers to
//             `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, the constant-rate
//             counter HIP's wall_clock64() reads (rate from
//             hipDeviceAttributeWallClockRate; 100 MHz on gfx1151).
//   nvidia -- llvm.nvvm.read.ptx.sreg.globaltimer: nanoseconds. This is the
//             non-profiler NVIDIA witness the sm_120 lane was missing.
// Both are emitted as `llvm.call_intrinsic`, so no extra dialect is needed.
//
// The start stamp is placed AFTER the entry block's last alloca, never before
// it. Its leader guard lowers to a branch that splits the entry block, and an
// alloca left behind the split is no longer in the entry block, so LLVM treats
// it as dynamic and will not promote it to registers. Measured on the gfx1151
// serial SSD kernel (2026-09-26): stamping at the block start turned a fully
// unrolled 2512-instruction kernel into a 924-instruction looped one that ran
// 2.4x faster than the clean image, so the "instrumented twin" timed a
// different program. Placing it after the allocas still left ~1230
// instructions -- any memory op ahead of that kernel's loops changed LLVM's
// optimization -- which is why timing callers bracket the unmodified kernel
// with an instrumented EMPTY marker (tessera.compiler.native_device_clock)
// instead of stamping the kernel itself. Everything before that point must be free of memory
// effects (constants, casts, views, the allocas themselves), otherwise the
// stamp would miss real work and the pass refuses.
//
// The span argument and the clock are recorded on the kernel as
// `tessera.device_clock_span`, so the package and its consumers can see that
// this image is the instrumented member of a clean/instrumented pair.
//===----------------------------------------------------------------------===//

#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace tessera {
namespace {

// Setup that may precede the start stamp: materializing constants, casting,
// viewing and assembling descriptors. Anything else -- even pure arithmetic
// whose result is used later -- is work the span must include (review of
// #856), so the pass refuses rather than stamping after it.
bool isSetupOp(Operation &op) {
  if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
    return false;
  return op.hasTrait<OpTrait::ConstantLike>() || isa<CastOpInterface>(op) ||
         isa<ViewLikeOpInterface>(op) ||
         isa<LLVM::UndefOp, LLVM::PoisonOp, LLVM::ZeroOp, LLVM::InsertValueOp,
             LLVM::AddrSpaceCastOp, memref::ViewOp, memref::MemorySpaceCastOp>(op);
}

constexpr StringLiteral kSpanAttr = "tessera.device_clock_span";

struct DeviceClockSpanPass
    : public PassWrapper<DeviceClockSpanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceClockSpanPass)
  DeviceClockSpanPass() = default;
  DeviceClockSpanPass(const DeviceClockSpanPass &other) : PassWrapper(other) {}
  Option<std::string> backend{*this, "backend",
                              llvm::cl::desc("rocm or nvidia"),
                              llvm::cl::init("")};

  StringRef getArgument() const final { return "tessera-device-clock-span"; }
  StringRef getDescription() const final {
    return "Stamp each GPU kernel's execution span with the target's "
           "constant-rate device clock into an appended span-buffer argument "
           "(kernel-side timing witness).";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
             scf::SCFDialect>();
  }

  // Semantic key (Decision #21a): the clock is chosen by backend and never
  // defaulted -- an unnamed backend is an error, not "rocm".
  FailureOr<StringRef> intrinsic(Operation *anchor) {
    if (backend == "rocm")
      return StringRef("llvm.readsteadycounter");
    if (backend == "nvidia")
      return StringRef("llvm.nvvm.read.ptx.sreg.globaltimer");
    anchor->emitError("TESSERA_DEVICE_CLOCK_BACKEND: device-clock span needs "
                      "backend=rocm or backend=nvidia, got '")
        << backend << "'";
    return failure();
  }

  LogicalResult instrument(gpu::GPUFuncOp kernel, StringRef clock) {
    if (kernel->hasAttr(kSpanAttr))
      return kernel.emitError("TESSERA_DEVICE_CLOCK_ALREADY_INSTRUMENTED: "
                              "kernel already carries a device-clock span");
    Region &body = kernel.getBody();
    if (!body.hasOneBlock())
      return kernel.emitError(
          "TESSERA_DEVICE_CLOCK_UNSTRUCTURED: device-clock span requires a "
          "single-block kernel body; a barrier before a return in divergent "
          "control flow would deadlock");
    SmallVector<gpu::ReturnOp> returns;
    kernel.walk([&](gpu::ReturnOp r) { returns.push_back(r); });
    Block &block = body.front();
    if (returns.size() != 1 || returns.front()->getBlock() != &block ||
        &block.back() != returns.front().getOperation())
      return kernel.emitError(
          "TESSERA_DEVICE_CLOCK_UNSTRUCTURED: device-clock span requires one "
          "gpu.return, as the final terminator of the kernel body; found ")
             << returns.size();

    MLIRContext *ctx = kernel.getContext();
    Location loc = kernel.getLoc();
    Type ptrTy = LLVM::LLVMPointerType::get(ctx, /*addressSpace=*/1);
    Type i64 = IntegerType::get(ctx, 64);
    unsigned index = kernel.getNumArguments();
    if (failed(kernel.insertArgument(index, ptrTy, DictionaryAttr(), loc)))
      return kernel.emitError("TESSERA_DEVICE_CLOCK_ABI: could not append the "
                              "span-buffer argument");
    Value span = kernel.getArgument(index);

    auto isLeader = [&](OpBuilder &b) -> Value {
      Value zero = arith::ConstantIndexOp::create(b, loc, 0);
      Value lead;
      for (gpu::Dimension d :
           {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
        Value id = gpu::ThreadIdOp::create(b, loc, d);
        Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, id, zero);
        lead = lead ? arith::AndIOp::create(b, loc, lead, eq).getResult() : eq;
      }
      return lead;
    };
    auto stamp = [&](OpBuilder &b, LLVM::AtomicBinOp op, int64_t slot) {
      Value lead = isLeader(b);
      auto guarded = scf::IfOp::create(b, loc, lead, /*withElseRegion=*/false);
      OpBuilder inner = guarded.getThenBodyBuilder();
      Value now = LLVM::CallIntrinsicOp::create(inner, loc, i64,
                                                inner.getStringAttr(clock),
                                                ValueRange{})
                      .getResult(0);
      Value address = span;
      if (slot != 0)
        address = LLVM::GEPOp::create(inner, 
            loc, ptrTy, i64, span, ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(slot)});
      LLVM::AtomicRMWOp::create(inner, loc, op, address, now,
                                      LLVM::AtomicOrdering::monotonic);
    };

    Operation *lastAlloca = nullptr;
    for (Operation &op : block)
      if (isa<LLVM::AllocaOp, memref::AllocaOp>(op))
        lastAlloca = &op;
    if (lastAlloca) {
      for (Operation &op : block) {
        // Only setup may precede the stamp (see isSetupOp): an scf.for or an
        // arith chain would run before it and be missed (reviews).
        if (!isa<LLVM::AllocaOp, memref::AllocaOp>(op) && !isSetupOp(op))
          return op.emitError(
              "TESSERA_DEVICE_CLOCK_ALLOCA_AFTER_WORK: an alloca follows an op "
              "that is not setup (constants, casts, views, descriptors); the "
              "start stamp cannot both precede the work and keep every alloca "
              "in the entry block");
        if (&op == lastAlloca)
          break;
      }
    }
    OpBuilder head = lastAlloca ? OpBuilder(block.getParentOp()->getContext())
                                : OpBuilder(&block, block.begin());
    if (lastAlloca)
      head.setInsertionPointAfter(lastAlloca);
    stamp(head, LLVM::AtomicBinOp::umin, 0);
    gpu::BarrierOp::create(head, loc);

    OpBuilder tail(returns.front());
    gpu::BarrierOp::create(tail, loc);
    stamp(tail, LLVM::AtomicBinOp::umax, 1);

    kernel->setAttr(kSpanAttr,
                    DictionaryAttr::get(
                        ctx, {NamedAttribute(StringAttr::get(ctx, "backend"),
                                             StringAttr::get(ctx, backend)),
                              NamedAttribute(StringAttr::get(ctx, "clock"),
                                             StringAttr::get(ctx, clock)),
                              NamedAttribute(StringAttr::get(ctx, "argument"),
                                             IntegerAttr::get(i64, index))}));
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<StringRef> clock = intrinsic(module);
    if (failed(clock))
      return signalPassFailure();
    SmallVector<gpu::GPUFuncOp> kernels;
    module.walk([&](gpu::GPUFuncOp f) {
      if (f.isKernel())
        kernels.push_back(f);
    });
    if (kernels.empty()) {
      module.emitError("TESSERA_DEVICE_CLOCK_NO_KERNEL: no gpu.func kernel to "
                       "instrument");
      return signalPassFailure();
    }
    for (gpu::GPUFuncOp kernel : kernels)
      if (failed(instrument(kernel, *clock)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDeviceClockSpanPass() {
  return std::make_unique<DeviceClockSpanPass>();
}

} // namespace tessera
