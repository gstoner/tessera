// Split, bufferized AD products to a bounded serial native GPU entry.
#include "Tessera/Transforms/Passes.h"
#include "Tessera/Dialect/Tile/TileDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
#include <functional>
#include <algorithm>
namespace tessera {
namespace {
struct NativeTapeToGPUPass : mlir::PassWrapper<NativeTapeToGPUPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeTapeToGPUPass)
  NativeTapeToGPUPass() = default;
  NativeTapeToGPUPass(const NativeTapeToGPUPass &other) : PassWrapper(other) {}
  Option<std::string> backend{*this, "backend", llvm::cl::desc("Private allocation lowering: nvidia or rocm"), llvm::cl::init("nvidia")};
  Option<bool> hostResultsOnly{*this, "host-results-only", llvm::cl::desc("Stop after checked capacity/result projection for the native CPU companion"), llvm::cl::init(false)};
  Option<bool> parallelRows{*this, "parallel-ann-rows", llvm::cl::desc("Assign proven independent ANN rows to GPU threads"), llvm::cl::init(false)};
  Option<bool> statusBuffer{*this, "status-buffer", llvm::cl::desc("Append a checked i64 status result for structured product assertions"), llvm::cl::init(false)};
  Option<int64_t> publicResultCapacity{*this, "public-result-capacity", llvm::cl::desc("Export generated scalar through rank-four AD results into checked capacity storage"), llvm::cl::init(0)};
  Option<int64_t> publicInputCapacity{*this, "public-input-capacity", llvm::cl::desc("Bind dynamic AD inputs to checked flat capacity and logical shape sidecars"), llvm::cl::init(0)};
  Option<bool> inputStatus{*this, "input-status", llvm::cl::desc("Require a successful incoming product status before any body effect"), llvm::cl::init(false)};
  Option<unsigned> inputStatusCount{*this, "input-status-count", llvm::cl::desc("Number of independently checked incoming statuses (1 through 8)"), llvm::cl::init(1)};
  llvm::StringRef getArgument() const final { return "tessera-native-tape-to-gpu"; }
  llvm::StringRef getDescription() const final { return "Materialize bounded bufferized AD/ANN/SSD products with proved temporary capacities"; }
  void getDependentDialects(mlir::DialectRegistry &r) const override {
    r.insert<tessera::tile::TesseraTileDialect,mlir::arith::ArithDialect,mlir::func::FuncDialect,mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,mlir::memref::MemRefDialect,mlir::scf::SCFDialect,mlir::math::MathDialect,mlir::cf::ControlFlowDialect>();
  }
  void runOnOperation() override {
    using namespace mlir;
    auto m=getOperation();
    auto reject=[&]() { m.emitError("native tape GPU requires an isolated static f32/f64 buffer product with bounded for/if control and at most 4096 temporary bytes"); signalPassFailure(); };
    if (backend!="nvidia" && backend!="rocm") return reject();
    SmallVector<func::FuncOp> fs(m.getOps<func::FuncOp>());
    bool ad=m->hasAttr("tessera.autodiff.product_abi") && m->hasAttr("tessera.autodiff.product_pair");
    bool ann=m->hasAttr("tessera.ann.source");
    bool ssd=m->hasAttr("tessera.ssd.source");
    bool publicResults=m->hasAttr("tessera.native_result_program");
    bool sourceState=m->hasAttr("tessera.source_state") && !ad && !ann && !ssd && !publicResults;
    if (sourceState && !publicResultCapacity) return reject();
    if (fs.size()!=1 || (static_cast<int>(ad)+static_cast<int>(ann)+static_cast<int>(ssd)+static_cast<int>(publicResults)+static_cast<int>(sourceState))!=1) return reject();
    auto f=fs[0];
    if (publicInputCapacity) {
      if (!ad || !statusBuffer || !publicResultCapacity || publicInputCapacity<1 || publicInputCapacity>1024 ||
          !f.getBody().hasOneBlock()) return reject();
      unsigned originalInputs=f.getNumArguments();
      for (unsigned i=0;i<originalInputs;++i) {
        auto type=dyn_cast<MemRefType>(f.getArgument(i).getType());
        if (!type || type.hasStaticShape()) continue;
        if (type.getRank()<1 || type.getRank()>4 || !type.getLayout().isIdentity() ||
            !(type.getElementType().isF32() || type.getElementType().isF64())) return reject();
        for (int64_t dim:type.getShape())
          if (!ShapedType::isDynamic(dim) && (dim<1 || dim>publicInputCapacity)) return reject();
        auto data=f.getArgument(i);
        unsigned shapeIndex=f.getNumArguments();
        auto shapeType=MemRefType::get({type.getRank()},IntegerType::get(m.getContext(),64));
        if (failed(f.insertArgument(shapeIndex,shapeType,DictionaryAttr{},f.getLoc()))) return reject();
        data.setType(MemRefType::get({publicInputCapacity},type.getElementType()));
        OpBuilder at=OpBuilder::atBlockBegin(&f.getBody().front());
        f.setArgAttr(i,"tessera.input_shape",at.getI64IntegerAttr(shapeIndex));
        auto one=arith::ConstantIndexOp::create(at,f.getLoc(),1);
        auto cap=arith::ConstantIndexOp::create(at,f.getLoc(),publicInputCapacity);
        SmallVector<OpFoldResult> sizes, strides(type.getRank());
        SmallVector<Value> extents;
        Value count=one;
        for (int64_t d=0;d<type.getRank();++d) {
          auto index=arith::ConstantIndexOp::create(at,f.getLoc(),d);
          auto raw=memref::LoadOp::create(at,f.getLoc(),f.getArgument(shapeIndex),ValueRange{index});
          auto extent=arith::IndexCastOp::create(at,f.getLoc(),at.getIndexType(),raw);
          auto fits=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::ule,extent,cap);
          cf::AssertOp::create(at,f.getLoc(),fits,at.getStringAttr("dynamic AD input exceeds capacity"));
          // Clamp has a syntactic unsigned range proof; the preceding guard
          // makes invalid sidecars fail before any input memory access.
          Value bounded=arith::MinUIOp::create(at,f.getLoc(),extent,cap);
          if (!type.isDynamicDim(d)) {
            auto expected=arith::ConstantIndexOp::create(at,f.getLoc(),type.getDimSize(d));
            auto same=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::eq,bounded,expected);
            cf::AssertOp::create(at,f.getLoc(),same,at.getStringAttr("dynamic AD input static dimension disagrees"));
            bounded=expected;
            sizes.push_back(at.getIndexAttr(type.getDimSize(d)));
          } else sizes.push_back(bounded);
          extents.push_back(bounded);
          count=arith::MulIOp::create(at,f.getLoc(),count,bounded);
        }
        auto fits=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::ule,count,cap);
        cf::AssertOp::create(at,f.getLoc(),fits,at.getStringAttr("dynamic AD input volume exceeds capacity"));
        Value stride=one;
        int64_t staticStride=1;
        for (int64_t d=type.getRank()-1;d>=0;--d) {
          strides[d]=staticStride>=0 ? OpFoldResult(at.getIndexAttr(staticStride)) : OpFoldResult(stride);
          stride=arith::MulIOp::create(at,f.getLoc(),stride,extents[d]);
          if (type.isDynamicDim(d) || staticStride<0) staticStride=-1;
          else staticStride*=type.getDimSize(d);
        }
        auto view=memref::ReinterpretCastOp::create(at,f.getLoc(),type,data,at.getIndexAttr(0),sizes,strides);
        data.replaceAllUsesExcept(view,view);
        SmallVector<memref::DimOp> dims;
        f.walk([&](memref::DimOp dim) { if (dim.getSource()==view) dims.push_back(dim); });
        for (auto dim:dims) {
          auto index=dim.getConstantIndex();
          if (!index) return reject();
          dim.replaceAllUsesWith(extents[*index]); dim.erase();
        }
      }
      f.setType(FunctionType::get(m.getContext(),f.getBody().front().getArgumentTypes(),f.getResultTypes()));
    }
    if (publicResultCapacity) {
      if (!(ad || sourceState) || !statusBuffer || inputStatus || publicResultCapacity<1 || publicResultCapacity>1024 ||
          !f.getBody().hasOneBlock() || !f.getNumResults()) return reject();
      auto ret=dyn_cast<func::ReturnOp>(f.getBody().front().getTerminator());
      if (!ret || ret.getNumOperands()!=f.getNumResults()) return reject();
      SmallVector<Value> returned(ret.getOperands());
      for (Value value:returned) {
        auto type=dyn_cast<MemRefType>(value.getType());
        if (!type || type.getRank()>4 ||
            !(type.getElementType().isF32() || type.getElementType().isF64() ||
              type.getElementType().isInteger(8) || type.getElementType().isInteger(64))) return reject();
        unsigned outIndex=f.getNumArguments();
        auto outputType=MemRefType::get({type.getRank()==0 ? 1 : int64_t(publicResultCapacity)},type.getElementType());
        auto shapeType=MemRefType::get({std::max<int64_t>(1,type.getRank())},IntegerType::get(m.getContext(),64));
        if (failed(f.insertArgument(outIndex,outputType,DictionaryAttr{},f.getLoc())) ||
            failed(f.insertArgument(outIndex+1,shapeType,DictionaryAttr{},f.getLoc()))) return reject();
        OpBuilder at(ret);
        f.setArgAttr(outIndex,"tessera.result_shape",at.getI64IntegerAttr(outIndex+1));
        if (type.getRank()==0) f.setArgAttr(outIndex,"tessera.result_scalar",at.getUnitAttr());
        auto zero=arith::ConstantIndexOp::create(at,f.getLoc(),0);
        auto one=arith::ConstantIndexOp::create(at,f.getLoc(),1);
        auto capacity=arith::ConstantIndexOp::create(at,f.getLoc(),publicResultCapacity);
        SmallVector<Value> extents;
        Value count=one;
        for (int64_t d=0;d<type.getRank();++d) {
          auto extent=memref::DimOp::create(at,f.getLoc(),value,d);
          auto fits=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::ule,extent,capacity);
          cf::AssertOp::create(at,f.getLoc(),fits,at.getStringAttr("generated AD extent exceeds public capacity"));
          count=arith::MulIOp::create(at,f.getLoc(),count,extent);
          extents.push_back(extent);
        }
        auto fits=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::ule,count,capacity);
        cf::AssertOp::create(at,f.getLoc(),fits,at.getStringAttr("generated AD result exceeds public capacity"));
        SmallVector<Value> indices;
        std::function<void(unsigned,Value)> copyDimension=[&](unsigned dim,Value ordinal) {
          if (dim==extents.size()) {
            auto element=memref::LoadOp::create(at,f.getLoc(),value,indices);
            memref::StoreOp::create(at,f.getLoc(),element,f.getArgument(outIndex),ValueRange{ordinal});
            return;
          }
          auto loop=scf::ForOp::create(at,f.getLoc(),zero,capacity,one);
          OpBuilder::InsertionGuard guard(at);
          at.setInsertionPointToStart(loop.getBody());
          auto active=arith::CmpIOp::create(at,f.getLoc(),arith::CmpIPredicate::ult,loop.getInductionVar(),extents[dim]);
          auto branch=scf::IfOp::create(at,f.getLoc(),active,false);
          at.setInsertionPointToStart(branch.thenBlock());
          auto next=arith::AddIOp::create(at,f.getLoc(),arith::MulIOp::create(at,f.getLoc(),ordinal,extents[dim]),loop.getInductionVar());
          indices.push_back(loop.getInductionVar());
          copyDimension(dim+1,next);
          indices.pop_back();
        };
        copyDimension(0,zero);
        if (type.getRank()==0) {
          auto scalarSize=arith::ConstantIntOp::create(at,f.getLoc(),1,64);
          memref::StoreOp::create(at,f.getLoc(),scalarSize,f.getArgument(outIndex+1),ValueRange{zero});
        }
        for (auto [d,extent]:llvm::enumerate(extents)) {
          auto index=arith::ConstantIndexOp::create(at,f.getLoc(),d);
          auto length=arith::IndexCastOp::create(at,f.getLoc(),at.getI64Type(),extent);
          memref::StoreOp::create(at,f.getLoc(),length,f.getArgument(outIndex+1),ValueRange{index});
        }
      }
      ret->setOperands(ValueRange{});
      f.setType(FunctionType::get(m.getContext(),f.getBody().front().getArgumentTypes(),TypeRange{}));
      m->setAttr("tessera.native_result_program",UnitAttr::get(m.getContext()));
      publicResults=true; ad=false;
    }
    if (!f.getBody().hasOneBlock() || f.getNumResults() || !isa<func::ReturnOp>(f.getBody().front().getTerminator())) return reject();
    if (publicResults) {
      if (!statusBuffer || inputStatus || parallelRows || m->hasAttr("tessera.native_result_abi")) return reject();
      llvm::json::Array arguments, results, inputs;
      llvm::SmallDenseSet<unsigned> outputSlots;
      for (unsigned i=0;i<f.getNumArguments();++i) {
        auto shapeSlot=f.getArgAttrOfType<IntegerAttr>(i,"tessera.result_shape");
        if (!shapeSlot) continue;
        auto dataType=dyn_cast<MemRefType>(f.getArgument(i).getType());
        int64_t j=shapeSlot.getInt();
        if (j<0 || j>=f.getNumArguments() || j==i || !dataType || dataType.getRank()!=1 || !dataType.hasStaticShape() ||
            !(dataType.getElementType().isF32() || dataType.getElementType().isF64() ||
              dataType.getElementType().isInteger(8) || dataType.getElementType().isInteger(64))) return reject();
        auto shapeType=dyn_cast<MemRefType>(f.getArgument(j).getType());
        if (!shapeType || shapeType.getRank()!=1 || !shapeType.hasStaticShape() ||
            shapeType.getDimSize(0)<1 || shapeType.getDimSize(0)>4 || !shapeType.getElementType().isInteger(64) ||
            outputSlots.contains(i) || outputSlots.contains(j)) return reject();
        int64_t rank=shapeType.getDimSize(0);
        outputSlots.insert(i); outputSlots.insert(j);
        llvm::json::Object result{{"data",i},{"shape",j},{"capacity",dataType.getDimSize(0)}};
        bool scalar=f.getArgAttr(i,"tessera.result_scalar")!=nullptr;
        if (scalar && (rank!=1 || dataType.getDimSize(0)!=1)) return reject();
        if (rank!=1 || scalar) result["rank"]=scalar ? 0 : rank;
        results.push_back(std::move(result));
        OpBuilder begin=OpBuilder::atBlockBegin(&f.getBody().front());
        auto missing=arith::ConstantIntOp::create(begin,f.getLoc(),-1,64);
        for (int64_t d=0;d<rank;++d) {
          auto index=arith::ConstantIndexOp::create(begin,f.getLoc(),d);
          memref::StoreOp::create(begin,f.getLoc(),missing,f.getArgument(j),ValueRange{index});
        }
        OpBuilder end(f.getBody().front().getTerminator());
        auto bound=arith::ConstantIntOp::create(end,f.getLoc(),dataType.getDimSize(0),64);
        Value count=arith::ConstantIntOp::create(end,f.getLoc(),1,64);
        for (int64_t d=0;d<rank;++d) {
          auto index=arith::ConstantIndexOp::create(end,f.getLoc(),d);
          auto extent=memref::LoadOp::create(end,f.getLoc(),f.getArgument(j),ValueRange{index});
          // Unsigned comparison rejects negative, missing and overflowing extents.
          auto ok=arith::CmpIOp::create(end,f.getLoc(),arith::CmpIPredicate::ule,extent,bound);
          cf::AssertOp::create(end,f.getLoc(),ok,end.getStringAttr("public result extent exceeds capacity"));
          count=arith::MulIOp::create(end,f.getLoc(),count,extent);
        }
        auto fits=arith::CmpIOp::create(end,f.getLoc(),arith::CmpIPredicate::ule,count,bound);
        cf::AssertOp::create(end,f.getLoc(),fits,end.getStringAttr("public result volume exceeds capacity"));
      }
      if (results.empty()) return reject();
      // Read-only argument roles must agree with executable stores, not just
      // the shape annotations. Unknown/forwarded write aliases refuse here.
      bool invalidWrite=false;
      auto writable=[&](Value value) {
        // Known view chains preserve the allocation's ownership. Do not infer
        // ownership through unknown producers or control-flow forwarding.
        for (unsigned depth=0;depth<32;++depth) {
          if (auto arg=dyn_cast<BlockArgument>(value))
            return arg.getOwner()==&f.getBody().front() && outputSlots.contains(arg.getArgNumber());
          if (isa_and_nonnull<memref::AllocOp,memref::AllocaOp>(value.getDefiningOp())) return true;
          if (auto view=value.getDefiningOp<memref::SubViewOp>()) value=view.getSource();
          else if (auto view=value.getDefiningOp<memref::CastOp>()) value=view.getSource();
          else if (auto view=value.getDefiningOp<memref::ReinterpretCastOp>()) value=view.getSource();
          else return false;
        }
        return false;
      };
      f.walk([&](Operation *op) {
        if (auto store=dyn_cast<memref::StoreOp>(op); store && !writable(store.getMemRef())) invalidWrite=true;
        if (auto copy=dyn_cast<memref::CopyOp>(op); copy && !writable(copy.getTarget())) invalidWrite=true;
        if (op->getName().getDialectNamespace()=="memref" &&
            !isa<memref::AllocOp,memref::AllocaOp,memref::DeallocOp,memref::GetGlobalOp,
                 memref::LoadOp,memref::StoreOp,memref::CopyOp,memref::DimOp,memref::SubViewOp,memref::CastOp,memref::ReinterpretCastOp>(op)) invalidWrite=true;
      });
      if (invalidWrite) return reject();
      for (unsigned i=0;i<f.getNumArguments();++i) {
        auto type=dyn_cast<MemRefType>(f.getArgument(i).getType());
        if (!type || !type.hasStaticShape()) return reject();
        if (auto slot=f.getArgAttrOfType<IntegerAttr>(i,"tessera.input_shape")) {
          auto j=slot.getInt();
          if (j<0 || j>=f.getNumArguments() || outputSlots.contains(i) || outputSlots.contains(j)) return reject();
          auto shapeType=dyn_cast<MemRefType>(f.getArgument(j).getType());
          if (!shapeType || shapeType.getRank()!=1 || !shapeType.hasStaticShape() ||
              shapeType.getDimSize(0)<1 || shapeType.getDimSize(0)>4 || !shapeType.getElementType().isInteger(64)) return reject();
          inputs.push_back(llvm::json::Object{{"data",i},{"shape",j},{"capacity",type.getDimSize(0)},{"rank",shapeType.getDimSize(0)}});
        }
        std::string storage; llvm::raw_string_ostream out(storage); type.getElementType().print(out);
        llvm::json::Array dims; for (auto d:type.getShape()) dims.push_back(d);
        arguments.push_back(llvm::json::Object{{"shape",std::move(dims)},{"storage",storage},{"writable",outputSlots.contains(i)}});
      }
      llvm::json::Object contract{{"schema",1},{"arguments",std::move(arguments)},{"results",std::move(results)}};
      if (!inputs.empty()) contract["inputs"]=std::move(inputs);
      auto encoded=llvm::formatv("{0}",llvm::json::Value(std::move(contract))).str();
      m->setAttr("tessera.native_result_abi",StringAttr::get(m.getContext(),encoded));
    }
    if (inputStatusCount < 1 || inputStatusCount > 8 || (!inputStatus && inputStatusCount != 1)) return reject();
    if (inputStatus && !statusBuffer) return reject();
    if (statusBuffer) {
      if ((!ad && !publicResults) || parallelRows) return reject();
      OpBuilder entry=OpBuilder::atBlockBegin(&f.getBody().front());
      auto statusType=MemRefType::get({1},entry.getI64Type());
      SmallVector<Value> dependencies;
      if (inputStatus) for (unsigned i=0; i<inputStatusCount; ++i) {
        if (failed(f.insertArgument(f.getNumArguments(),statusType,DictionaryAttr{},f.getLoc()))) return reject();
        dependencies.push_back(f.getArguments().back());
      }
      if (failed(f.insertArgument(f.getNumArguments(),statusType,DictionaryAttr{},f.getLoc()))) return reject();
      auto status=f.getArguments().back();
      auto zero=arith::ConstantIndexOp::create(entry,f.getLoc(),0);
      auto success=arith::ConstantIntOp::create(entry,f.getLoc(),0,64);
      memref::StoreOp::create(entry,f.getLoc(),success,status,ValueRange{zero});
      for (Value dependency : dependencies) {
        auto incoming=memref::LoadOp::create(entry,f.getLoc(),dependency,ValueRange{zero});
        auto ok=arith::CmpIOp::create(entry,f.getLoc(),arith::CmpIPredicate::eq,incoming,success);
        cf::AssertOp::create(entry,f.getLoc(),ok,entry.getStringAttr("upstream product failed"));
        m->setAttr("tessera.autodiff.input_status",entry.getStringAttr("guard-v1"));
        if (inputStatusCount > 1) m->setAttr("tessera.autodiff.input_status_count",entry.getI64IntegerAttr(inputStatusCount));
      }
      // A nested failure must suppress the enclosing suffix and every later
      // loop iteration. Loop-carried scalars keep their previous value while
      // failed branches yield inert scalars; no failed result is exposed.
      auto assertStatus=[&](OpBuilder &at, Location loc) {
        auto value=memref::LoadOp::create(at,loc,status,ValueRange{zero});
        auto ok=arith::CmpIOp::create(at,loc,arith::CmpIPredicate::eq,value,success);
        cf::AssertOp::create(at,loc,ok,at.getStringAttr("nested product failed"));
      };
      std::function<FailureOr<bool>(Block &)> guardBlock;
      guardBlock=[&](Block &block) -> FailureOr<bool> {
        SmallVector<Operation *> original;
        for (auto &op:block.without_terminator()) original.push_back(&op);
        bool contains=false;
        for (auto *op:original) {
          if (isa<cf::AssertOp>(op)) contains=true;
          bool child=false;
          for (auto &region:op->getRegions()) {
            if (region.empty()) continue;
            if (!isa<scf::ForOp,scf::IfOp>(op) || !region.hasOneBlock()) return failure();
            auto result=guardBlock(region.front());
            if (failed(result)) return failure();
            child|=*result;
          }
          if (child) {
            OpBuilder after(op); after.setInsertionPointAfter(op);
            assertStatus(after,op->getLoc());
            contains=true;
          }
        }
        if (!contains) return false;
        auto *terminator=block.getTerminator();
        SmallVector<Value> fallback;
        if (auto loop=dyn_cast<scf::ForOp>(block.getParentOp())) {
          llvm::append_range(fallback,loop.getRegionIterArgs());
        } else {
          OpBuilder before=OpBuilder::atBlockBegin(&block);
          for (auto type:terminator->getOperandTypes()) {
            if (!type.isIntOrIndexOrFloat()) return failure();
            fallback.push_back(arith::ConstantOp::create(before,terminator->getLoc(),type,before.getZeroAttr(type)));
          }
        }
        if (fallback.size()!=terminator->getNumOperands()) return failure();
        // This entry check is needed even when the original assertion occurs
        // after a load in the loop body: later iterations must do no work.
        if (&block!=&f.getBody().front()) {
          OpBuilder begin=OpBuilder::atBlockBegin(&block);
          if (!fallback.empty())
            if (auto *definition=fallback.back().getDefiningOp(); definition && definition->getBlock()==&block)
              begin.setInsertionPointAfter(definition);
          assertStatus(begin,terminator->getLoc());
        }
        SmallVector<cf::AssertOp> guards(block.getOps<cf::AssertOp>());
        for (auto guard:llvm::reverse(guards)) {
          SmallVector<Value> values(terminator->getOperands());
          SmallVector<Type> types(terminator->getOperandTypes());
          OpBuilder at(guard);
          auto branch=scf::IfOp::create(at,guard.getLoc(),types,guard.getArg(),true);
          auto *next=guard->getNextNode();
          // The new branch precedes guard; move only its original suffix.
          while (next!=terminator) {
            auto *following=next->getNextNode();
            next->moveBefore(branch.thenBlock(),branch.thenBlock()->end());
            next=following;
          }
          // Result-free scf.if creates empty yields; place them after the suffix.
          if (types.empty()) {
            auto yield=cast<scf::YieldOp>(&branch.thenBlock()->front());
            yield->moveBefore(branch.thenBlock(),branch.thenBlock()->end());
          } else {
            at.setInsertionPointToEnd(branch.thenBlock());
            scf::YieldOp::create(at,guard.getLoc(),values);
          }
          at.setInsertionPointToStart(branch.elseBlock());
          auto failureValue=arith::ConstantIntOp::create(at,guard.getLoc(),1,64);
          memref::StoreOp::create(at,guard.getLoc(),failureValue,status,ValueRange{zero});
          if (!types.empty()) scf::YieldOp::create(at,guard.getLoc(),fallback);
          terminator->setOperands(branch.getResults());
          guard.erase();
        }
        return true;
      };
      if (failed(guardBlock(f.getBody().front()))) return reject();
      m->setAttr("tessera.autodiff.gpu_status",entry.getStringAttr("guard-v1"));
    }
    if (hostResultsOnly) {
      if (!publicResults || !publicResultCapacity || !statusBuffer || parallelRows || inputStatus) return reject();
      m->removeAttr("tessera.autodiff.gpu_status");
      m->setAttr("tessera.native_result_status",StringAttr::get(m.getContext(),"guard-v1"));
      return;
    }
    auto admissible=[](Type t) {
      auto mt=dyn_cast<MemRefType>(t);
      if (!mt || !mt.hasStaticShape() || !(mt.getElementType().isF32() || mt.getElementType().isF64() || mt.getElementType().isInteger(8) || mt.getElementType().isInteger(64) || mt.getElementType().isInteger(1) || mt.getElementType().isIndex()) || !mt.getLayout().isIdentity() || mt.getMemorySpace()) return false;
      int64_t count=1;
      for (auto d:mt.getShape()) {
        if (d<=0 || d>1024 || count>1024/d) return false;
        count*=d;
      }
      return true;
    };
    auto elementBytes=[](Type t) -> int64_t { return t.isIndex() ? 8 : (t.getIntOrFloatBitWidth()+7)/8; };
    for (auto t:f.getArgumentTypes()) if (!admissible(t)) return reject();
    // Derive replay-loop capacity from SSA arithmetic over enclosing bounded
    // induction variables. Never trust a user-written max_iters attribute or
    // a loaded counter. Intervals stay small so endpoint arithmetic cannot wrap.
    using Interval = std::pair<int64_t,int64_t>;
    std::function<std::optional<Interval>(Value,unsigned)> range;
    range = [&](Value value,unsigned depth) -> std::optional<Interval> {
      if (depth>64 || !value.getType().isIndex()) return std::nullopt;
      if (auto clamp=value.getDefiningOp<arith::MinUIOp>()) {
        for (Value operand:clamp.getOperands()) {
          APInt bound;
          if (matchPattern(operand,m_ConstantInt(&bound)) && bound.isSignedIntN(64) &&
              bound.getSExtValue()>=0 && bound.getSExtValue()<=1024)
            return Interval{0,bound.getSExtValue()};
        }
      }
      if (auto select=value.getDefiningOp<arith::SelectOp>()) {
        auto yes=range(select.getTrueValue(),depth+1), no=range(select.getFalseValue(),depth+1);
        if (!yes || !no) return std::nullopt;
        return Interval{std::min(yes->first,no->first),std::max(yes->second,no->second)};
      }
      APInt constant;
      if (matchPattern(value,m_ConstantInt(&constant))) {
        if (!constant.isSignedIntN(64)) return std::nullopt;
        auto v=constant.getSExtValue();
        if (v < -1024 || v > 1024) return std::nullopt;
        return Interval{v,v};
      }
      if (auto arg=dyn_cast<BlockArgument>(value)) {
        auto owner=dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (!owner || owner->hasAttr("unsignedCmp") || value!=owner.getInductionVar()) return std::nullopt;
        auto low=range(owner.getLowerBound(),depth+1);
        auto high=range(owner.getUpperBound(),depth+1);
        auto step=range(owner.getStep(),depth+1);
        if (!low || !high || !step || step->first<=0 || step->first!=step->second)
          return std::nullopt;
        return Interval{low->first,std::max(low->first,high->second-1)};
      }
      auto *op=value.getDefiningOp();
      if (!op) return std::nullopt;
      if (op->getNumOperands()!=2) return std::nullopt;
      auto a=range(op->getOperand(0),depth+1), b=range(op->getOperand(1),depth+1);
      if (!a || !b) return std::nullopt;
      Interval result;
      if (isa<arith::AddIOp>(op)) result={a->first+b->first,a->second+b->second};
      else if (isa<arith::DivUIOp>(op)) {
        // Unsigned monotone division is safe only for proven nonnegative
        // numerators and strictly positive denominators. No zero/negative
        // divisor or wrapped signed interval may establish allocation bounds.
        if (a->first < 0 || b->first <= 0) return std::nullopt;
        result={a->first/b->second,a->second/b->first};
      }
      else if (isa<arith::SubIOp>(op)) result={a->first-b->second,a->second-b->first};
      else if (isa<arith::MulIOp>(op)) {
        int64_t products[]={a->first*b->first,a->first*b->second,a->second*b->first,a->second*b->second};
        result={*std::min_element(products,products+4),*std::max_element(products,products+4)};
      } else return std::nullopt;
      if (result.first < -1024 || result.second > 1024) return std::nullopt;
      return result;
    };
    auto tripCapacity=[&](scf::ForOp loop) -> std::optional<int64_t> {
      if (loop->hasAttr("unsignedCmp")) return std::nullopt;
      auto low=range(loop.getLowerBound(),0), high=range(loop.getUpperBound(),0), step=range(loop.getStep(),0);
      if (!low || !high || !step || low->first<0 ||
          step->first!=step->second || step->first<=0) return std::nullopt;
      auto count=std::max<int64_t>(1,(high->second-low->first+step->first-1)/step->first);
      if (count>1024) return std::nullopt;
      return count;
    };
    // Physical capacity is proved from SSA ranges; logical dimensions stay
    // dynamic in each view and copy. Loaded extents and unknown aliases refuse.
    auto allocationShape=[&](Operation *op) -> std::optional<SmallVector<int64_t>> {
      auto type=dyn_cast<MemRefType>(op->getResult(0).getType());
      if (!type || !type.getLayout().isIdentity() || type.getMemorySpace()) return std::nullopt;
      SmallVector<int64_t> shape(type.getShape());
      if (!type.hasStaticShape()) {
        auto alloc=dyn_cast<memref::AllocOp>(op);
        if (!alloc) return std::nullopt;
        unsigned index=0;
        for (auto &dim:shape) if (ShapedType::isDynamic(dim)) {
          auto bounds=range(alloc.getDynamicSizes()[index++],0);
          if (!bounds || bounds->first<0 || bounds->second<=0) return std::nullopt;
          dim=bounds->second;
        }
      }
      // Per-axis intervals can overestimate a guarded joint volume (e.g.
      // dimensions each <= 64 whose product is <= 64). Only a dominating
      // then-edge with the exact SSA product can tighten physical storage.
      int64_t count=1;
      for (int64_t dim:shape) {
        if (dim<1 || dim>1024 || count>(INT64_MAX/dim)) return std::nullopt;
        count*=dim;
      }
      if (auto alloc=dyn_cast<memref::AllocOp>(op); alloc && !type.hasStaticShape()) {
        SmallVector<Value> expected(alloc.getDynamicSizes());
        int64_t fixed=1;
        for (int64_t dim:type.getShape()) if (!ShapedType::isDynamic(dim)) fixed*=dim;
        for (Operation *child=op,*parent=op->getParentOp(); parent && parent!=f.getOperation();
             child=parent,parent=parent->getParentOp()) {
          auto branch=dyn_cast<scf::IfOp>(parent);
          if (!branch || child->getParentRegion()!=&branch.getThenRegion()) continue;
          auto cmp=branch.getCondition().getDefiningOp<arith::CmpIOp>();
          if (!cmp || cmp.getPredicate()!=arith::CmpIPredicate::ule) continue;
          APInt upper;
          if (!matchPattern(cmp.getRhs(),m_ConstantInt(&upper)) || !upper.isSignedIntN(64) ||
              upper.getSExtValue()<1 || upper.getSExtValue()>1024) continue;
          SmallVector<Value> remaining(expected);
          int64_t coefficient=1;
          std::function<bool(Value,unsigned)> matchProduct=[&](Value v,unsigned depth) {
            if (depth>32) return false;
            // Match an extent before descending into its defining arithmetic.
            auto found=llvm::find(remaining,v);
            if (found!=remaining.end()) { remaining.erase(found); return true; }
            APInt constant;
            if (matchPattern(v,m_ConstantInt(&constant))) {
              if (!constant.isSignedIntN(64) || constant.getSExtValue()<1 ||
                  constant.getSExtValue()>1024 || coefficient>1024/constant.getSExtValue()) return false;
              coefficient*=constant.getSExtValue(); return true;
            }
            auto mul=v.getDefiningOp<arith::MulIOp>();
            return mul && matchProduct(mul.getLhs(),depth+1) && matchProduct(mul.getRhs(),depth+1);
          };
          if (matchProduct(cmp.getLhs(),0) && remaining.empty() && coefficient==fixed)
            count=std::min(count,upper.getSExtValue());
        }
      }
      auto capacity=MemRefType::get({count},type.getElementType());
      if (!admissible(capacity)) return std::nullopt;
      return SmallVector<int64_t>{count};
    };
    auto logicalShape=[&](Value value) -> std::optional<SmallVector<OpFoldResult>> {
      auto type=dyn_cast<MemRefType>(value.getType());
      if (!type) return std::nullopt;
      if (auto view=value.getDefiningOp<memref::SubViewOp>()) {
        if (view.getSourceType().getRank()!=type.getRank()) return std::nullopt;
        return view.getMixedSizes();
      }
      if (auto view=value.getDefiningOp<memref::ReinterpretCastOp>())
        return view.getMixedSizes();
      auto alloc=value.getDefiningOp<memref::AllocOp>();
      if (!alloc && !type.hasStaticShape()) return std::nullopt;
      SmallVector<OpFoldResult> sizes;
      unsigned dynamic=0;
      Builder builder(m.getContext());
      for (int64_t n:type.getShape())
        sizes.push_back(ShapedType::isDynamic(n) ? OpFoldResult(alloc.getDynamicSizes()[dynamic++])
                                                : OpFoldResult(builder.getIndexAttr(n)));
      return sizes;
    };
    bool bad=false;
    int64_t bytes=0;
    f.walk([&](Operation *op) {
      auto dialect=op->getName().getDialectNamespace();
      if (op!=f.getOperation() && dialect!="arith" && dialect!="math" && dialect!="memref" && dialect!="scf" && !isa<func::ReturnOp>(op)) bad=true;
      if (isa<scf::WhileOp,func::CallOp,memref::DeallocOp>(op)) bad=true;
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        auto source=cast<MemRefType>(copy.getSource().getType());
        auto target=cast<MemRefType>(copy.getTarget().getType());
        // A dynamic copy must have identical logical shape SSA on both
        // allocations; capacity equality is not a logical shape proof.
        if (source.getShape()!=target.getShape()) bad=true;
        if (!source.hasStaticShape() || !target.hasStaticShape()) {
          auto a=logicalShape(copy.getSource()), b=logicalShape(copy.getTarget());
          if (!a || !b || a->size()!=b->size()) bad=true;
          else for (auto [left,right]:llvm::zip(*a,*b)) {
            if (left==right) continue;
            auto lhs=dyn_cast<Value>(left), rhs=dyn_cast<Value>(right);
            bool proved=false;
            if (lhs && rhs) for (Operation *child=op,*parent=op->getParentOp(); parent && parent!=f.getOperation(); child=parent,parent=parent->getParentOp()) {
              auto branch=dyn_cast<scf::IfOp>(parent);
              if (!branch || child->getParentRegion()!=&branch.getThenRegion()) continue;
              auto cmp=branch.getCondition().getDefiningOp<arith::CmpIOp>();
              if (cmp && cmp.getPredicate()==arith::CmpIPredicate::eq &&
                  ((cmp.getLhs()==lhs && cmp.getRhs()==rhs) || (cmp.getLhs()==rhs && cmp.getRhs()==lhs))) proved=true;
            }
            if (!proved) bad=true;
          }
        }
      }
      if (op->getNumRegions() && op!=f.getOperation() && !isa<scf::ForOp,scf::IfOp>(op)) bad=true;
      if (auto loop=dyn_cast<scf::ForOp>(op); loop && !tripCapacity(loop)) bad=true;
      if (!isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp>(op)) return;
      auto capacity=allocationShape(op);
      if (!capacity) { bad=true; return; }
      int64_t n=elementBytes(cast<MemRefType>(op->getResult(0).getType()).getElementType());
      for (int64_t dim:*capacity) n*=dim;
      for (Operation *parent=op->getParentOp();parent && parent!=f.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) {
          auto capacity=tripCapacity(loop);
          if (!capacity || n>4096 / *capacity) { bad=true; return; }
          int64_t trip=*capacity;
          n*=trip;
        }
      bytes+=n;
      if (bytes>4096 && !parallelRows) bad=true;
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto value=global ? dyn_cast_or_null<DenseElementsAttr>(global.getInitialValueAttr()) : DenseElementsAttr();
        if (!global || !global.getConstant() || !value) bad=true;
      }
    });
    int64_t rowCount=0;
    if (parallelRows) {
      if (!ann || f.getNumArguments()!=2) return reject();
      auto input=cast<MemRefType>(f.getArgument(0).getType());
      auto output=cast<MemRefType>(f.getArgument(1).getType());
      if (input.getRank()!=2 || output.getRank()<1 ||
          input.getDimSize(0)<2 || input.getDimSize(0)>64 ||
          output.getDimSize(0)!=input.getDimSize(0)) return reject();
      rowCount=input.getDimSize(0);
      auto rowLoop=[&](scf::ForOp loop) {
        APInt low,high,step;
        return loop.getNumRegionIterArgs()==0 &&
               matchPattern(loop.getLowerBound(),m_ConstantInt(&low)) && low.isZero() &&
               matchPattern(loop.getUpperBound(),m_ConstantInt(&high)) && high==rowCount &&
               matchPattern(loop.getStep(),m_ConstantInt(&step)) && step.isOne();
      };
      auto owned=[&](Value memref) {
        if (memref==f.getArgument(0) || memref==f.getArgument(1)) return true;
        return isa_and_nonnull<memref::AllocOp,memref::AllocaOp>(memref.getDefiningOp());
      };
      auto access=[&](Operation *op,Value memref,ValueRange indices,bool write) {
        if (memref.getDefiningOp<memref::GetGlobalOp>()) return !write;
        auto type=cast<MemRefType>(memref.getType());
        if (!owned(memref) || indices.empty() || type.getDimSize(0)!=rowCount ||
            (write && memref==f.getArgument(0))) return false;
        Operation *outer=op;
        while (outer->getParentOp()!=f.getOperation()) outer=outer->getParentOp();
        auto loop=dyn_cast<scf::ForOp>(outer);
        return loop && rowLoop(loop) && indices.front()==loop.getInductionVar();
      };
      f.walk([&](Operation *op) {
        // Memory operations not covered by the row-access proof (atomics,
        // alias construction, DMA, etc.) must not bypass it.
        if (op->getName().getDialectNamespace()=="memref" &&
            !isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp,
                 memref::LoadOp,memref::StoreOp,memref::CopyOp,memref::DimOp>(op)) bad=true;
        if (auto loop=dyn_cast<scf::ForOp>(op); loop && loop->getParentOp()==f.getOperation() && !rowLoop(loop)) bad=true;
        if (auto load=dyn_cast<memref::LoadOp>(op); load && !access(op,load.getMemRef(),load.getIndices(),false)) bad=true;
        if (auto store=dyn_cast<memref::StoreOp>(op); store && !access(op,store.getMemRef(),store.getIndices(),true)) bad=true;
        if (auto copy=dyn_cast<memref::CopyOp>(op)) {
          auto type=cast<MemRefType>(copy.getSource().getType());
          if (op->getParentOp()!=f.getOperation() || type.getRank()<1 || type.getDimSize(0)!=rowCount ||
              !owned(copy.getTarget()) || copy.getTarget()==f.getArgument(0) ||
              (!owned(copy.getSource()) && !copy.getSource().getDefiningOp<memref::GetGlobalOp>())) bad=true;
        }
      });
    }
    // Only entry-owned static mutable row buffers can be compacted. Constants
    // remain complete; nested allocation generations retain their slot proof.
    auto rowPrivate = [&](Operation *op) {
      if (!parallelRows || op->getParentOp()!=f.getOperation() ||
          !isa<memref::AllocOp,memref::AllocaOp>(op)) return false;
      auto type=cast<MemRefType>(op->getResult(0).getType());
      return type.hasStaticShape() && type.getRank()>0 && type.getDimSize(0)==rowCount;
    };
    llvm::DenseSet<Value> compactRows;
    if (parallelRows && !bad) {
      f.walk([&](Operation *op) {
        if (!rowPrivate(op)) return;
        auto type=cast<MemRefType>(op->getResult(0).getType());
        int64_t full=type.getNumElements()*elementBytes(type.getElementType());
        bytes-=full-full/rowCount;
        compactRows.insert(op->getResult(0));
      });
      if (bytes>4096) bad=true;
    }
    if (bad) return reject();
    std::string text; llvm::raw_string_ostream os(text);
    os<<"module { gpu.module @native_tape { gpu.func @product(";
    for (unsigned i=0;i<f.getNumArguments();++i) os<<"%p"<<i<<": !llvm.ptr<1>, ";
    os<<"%scratch: index) kernel {\n";
    for (auto [i,t]:llvm::enumerate(f.getArgumentTypes())) {
      auto mt=cast<MemRefType>(t); auto rank=mt.getRank();
      std::string desc; llvm::raw_string_ostream ds(desc);
      ds<<"!llvm.struct<(ptr, ptr, i64";
      if (rank) ds<<", array<"<<rank<<" x i64>, array<"<<rank<<" x i64>";
      ds<<")>"; ds.flush();
      os<<"%raw"<<i<<" = llvm.addrspacecast %p"<<i<<" : !llvm.ptr<1> to !llvm.ptr\n";
      os<<"%d"<<i<<"_0 = llvm.mlir.undef : "<<desc<<"\n";
      unsigned version=0;
      auto insert=[&](StringRef value,std::string position) {
        os<<"%d"<<i<<"_"<<(version+1)<<" = llvm.insertvalue "<<value<<", %d"<<i<<"_"<<version<<"["<<position<<"] : "<<desc<<"\n"; ++version;
      };
      insert("%raw"+std::to_string(i),"0"); insert("%raw"+std::to_string(i),"1");
      os<<"%z"<<i<<" = llvm.mlir.constant(0 : i64) : i64\n"; insert("%z"+std::to_string(i),"2");
      int64_t stride=1;
      for (int d=rank-1;d>=0;--d) {
        auto prefix="%shape"+std::to_string(i)+"_"+std::to_string(d);
        os<<prefix<<" = llvm.mlir.constant("<<mt.getDimSize(d)<<" : i64) : i64\n";
        insert(prefix,"3, "+std::to_string(d));
        os<<prefix<<"s = llvm.mlir.constant("<<stride<<" : i64) : i64\n";
        insert(prefix+"s","4, "+std::to_string(d)); stride*=mt.getDimSize(d);
      }
      os<<"%m"<<i<<" = builtin.unrealized_conversion_cast %d"<<i<<"_"<<version<<" : "<<desc<<" to "<<mt<<"\n";
    }
    os<<"%marker = memref.alloca(%scratch) : memref<?xf32>\n\"tile.alloc_shared\"(%marker) : (memref<?xf32>) -> ()\ngpu.return\n} } }"; os.flush();
    auto gpu=parseSourceString<ModuleOp>(text,m.getContext());
    if (!gpu) return reject();
    auto deviceModule=*gpu->getOps<gpu::GPUModuleOp>().begin();
    auto kernel=*deviceModule.getOps<gpu::GPUFuncOp>().begin();
    IRMapping mapping; unsigned argument=0;
    kernel.walk([&](UnrealizedConversionCastOp cast){mapping.map(f.getArgument(argument++),cast.getResult(0));});
    OpBuilder b(kernel.getBody().front().getTerminator());
    for (auto &op:f.getBody().front().without_terminator()) b.clone(op,mapping);
    llvm::DenseSet<Value> compactClones;
    for (Value value:compactRows) compactClones.insert(mapping.lookup(value));
    SmallVector<Operation *> memory;
    kernel.walk([&](Operation *op){if (isa<memref::AllocOp,memref::GetGlobalOp,memref::CopyOp>(op) || (isa<memref::AllocaOp>(op) && cast<MemRefType>(op->getResult(0).getType()).hasStaticShape())) memory.push_back(op);});
    llvm::stable_sort(memory, [](Operation *a, Operation *b) {
      return isa<memref::CopyOp>(a) && !isa<memref::CopyOp>(b);
    });
    for (auto *op:memory) {
      OpBuilder at(op); auto loc=op->getLoc();
      auto loopNest=[&](MemRefType type,auto leaf) {
        SmallVector<Value> indices;
        std::function<void(int)> recurse=[&](int dim) {
          if (dim==type.getRank()) { leaf(at,indices); return; }
          auto zero=arith::ConstantIndexOp::create(at,loc,0), one=arith::ConstantIndexOp::create(at,loc,1);
          Value end;
          if (type.isDynamicDim(dim))
            end=memref::DimOp::create(at,loc,cast<memref::CopyOp>(op).getSource(),dim);
          else end=arith::ConstantIndexOp::create(at,loc,type.getDimSize(dim));
          auto loop=scf::ForOp::create(at,loc,zero,end,one);
          OpBuilder::InsertionGuard guard(at); at.setInsertionPointToStart(loop.getBody());
          indices.push_back(loop.getInductionVar()); recurse(dim+1); indices.pop_back();
        }; recurse(0);
      };
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        loopNest(cast<MemRefType>(copy.getSource().getType()),[&](OpBuilder &builder,ValueRange indices) {
          auto v=memref::LoadOp::create(builder,loc,copy.getSource(),indices);
          memref::StoreOp::create(builder,loc,v,copy.getTarget(),indices);
        }); op->erase(); continue;
      }
      auto type=cast<MemRefType>(op->getResult(0).getType());
      if (compactClones.contains(op->getResult(0))) {
        Value value=op->getResult(0);
        SmallVector<Operation *> users(value.getUsers());
        Value zero=arith::ConstantIndexOp::create(at,loc,0);
        for (Operation *user:users) {
          if (auto load=dyn_cast<memref::LoadOp>(user))
            load.getIndicesMutable().slice(0,1).assign(zero);
          else if (auto store=dyn_cast<memref::StoreOp>(user))
            store.getIndicesMutable().slice(0,1).assign(zero);
          else if (auto dim=dyn_cast<memref::DimOp>(user)) {
            auto index=dim.getConstantIndex();
            if (!index) return reject();
            OpBuilder before(dim);
            auto extent=arith::ConstantIndexOp::create(before,dim.getLoc(),type.getDimSize(*index));
            dim.replaceAllUsesWith(extent.getResult());
            dim.erase();
          } else return reject();
        }
        SmallVector<int64_t> shape(type.getShape()); shape[0]=1;
        type=MemRefType::get(shape,type.getElementType());
        value.setType(type);
      }
      // A GPU static alloca is not a fresh allocation on each loop trip.
      // Give every syntactic allocation an entry-owned byte array and derive
      // a distinct slot from the complete enclosing iteration path.
      SmallVector<scf::ForOp> loops;
      int64_t slots=1;
      for (auto *parent=op->getParentOp();parent!=kernel.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) loops.push_back(loop);
      Value slot=arith::ConstantIndexOp::create(at,loc,0);
      for (auto loop:llvm::reverse(loops)) {
        // Cloning preserves the validated SSA range relationships.
        int64_t trip=*tripCapacity(loop);
        slots*=trip;
        auto count=arith::ConstantIndexOp::create(at,loc,trip);
        auto delta=arith::SubIOp::create(at,loc,loop.getInductionVar(),loop.getLowerBound());
        auto ordinal=arith::DivUIOp::create(at,loc,delta,loop.getStep());
        slot=arith::AddIOp::create(at,loc,arith::MulIOp::create(at,loc,slot,count),ordinal);
      }
      auto capacity=allocationShape(op);
      if (!capacity) return reject();
      int64_t slotBytes=elementBytes(type.getElementType());
      for (int64_t dim:*capacity) slotBytes*=dim;
      SmallVector<Value> dynamicSizes;
      if (auto alloc=dyn_cast<memref::AllocOp>(op)) llvm::append_range(dynamicSizes,alloc.getDynamicSizes());
      OpBuilder entry=OpBuilder::atBlockBegin(&kernel.getBody().front());
      // AMDGPU requires explicit private FrameIndex addressing. NVVM owns
      // conversion of generic allocas to its local address representation;
      // forcing AMD's private descriptor representation through NVVM is unsafe.
      Attribute space=backend=="rocm" ? Attribute(entry.getI64IntegerAttr(5)) : Attribute();
      auto root=memref::AllocaOp::create(entry,loc,MemRefType::get({slots*slotBytes},entry.getI8Type(),MemRefLayoutAttrInterface{},space));
      root->setAttr("alignment",entry.getI64IntegerAttr(64));
      auto stride=arith::ConstantIndexOp::create(at,loc,slotBytes);
      auto offset=arith::MulIOp::create(at,loc,slot,stride);
      auto privateType=MemRefType::get(type.getShape(),type.getElementType(),MemRefLayoutAttrInterface{},space);
      auto view=memref::ViewOp::create(at,loc,privateType,root,offset,dynamicSizes);
      Value local=view;
      if (space) local=memref::MemorySpaceCastOp::create(at,loc,type,view);
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto values=cast<DenseElementsAttr>(global.getInitialValueAttr());
        // Constants are bounded by the same 4096-byte resource check. Preserve
        // every element; ANN weights need not be splats.
        int64_t ordinal=0;
        for (auto value:values.getValues<Attribute>()) {
          SmallVector<Value> indices(type.getRank());
          int64_t remaining=ordinal++;
          for (int d=type.getRank()-1;d>=0;--d) {
            indices[d]=arith::ConstantIndexOp::create(at,loc,remaining%type.getDimSize(d));
            remaining/=type.getDimSize(d);
          }
          auto v=arith::ConstantOp::create(at,loc,type.getElementType(),cast<TypedAttr>(value));
          memref::StoreOp::create(at,loc,v,local,indices);
        }
      }
      op->getResult(0).replaceAllUsesWith(local); op->erase();
    }
    if (parallelRows) {
      OpBuilder at=OpBuilder::atBlockBegin(&kernel.getBody().front());
      auto loc=kernel.getLoc();
      Value row=gpu::ThreadIdOp::create(at,loc,gpu::Dimension::x);
      Value one=arith::ConstantIndexOp::create(at,loc,1);
      Value end=arith::AddIOp::create(at,loc,row,one);
      // Every mutable access was proved row-local before cloning. This also
      // narrows the top-level loops expanded from admitted memref.copy ops:
      // their full-buffer side keeps the thread row while compacted private
      // accesses use zero. Constants remain complete private/read-only arrays.
      for (auto loop:kernel.getBody().front().getOps<scf::ForOp>()) {
        loop.getLowerBoundMutable().assign(row);
        loop.getUpperBoundMutable().assign(end);
      }
      kernel->setAttr("known_block_size",at.getDenseI32ArrayAttr({static_cast<int32_t>(rowCount),1,1}));
    }
    m.getBody()->clear();
    m.getBody()->getOperations().splice(m.getBody()->end(),gpu->getBody()->getOperations());
    m->setAttr("tessera.autodiff.temporary_bytes",IntegerAttr::get(IntegerType::get(m.getContext(),64),bytes));
  }
};
}
std::unique_ptr<mlir::Pass> createNativeTapeToGPUPass() { return std::make_unique<NativeTapeToGPUPass>(); }
}
