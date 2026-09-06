// Split, bufferized AD products to a bounded serial native GPU entry.
#include "Tessera/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
namespace tessera {
namespace {
struct NativeTapeToGPUPass : mlir::PassWrapper<NativeTapeToGPUPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NativeTapeToGPUPass)
  NativeTapeToGPUPass() = default;
  NativeTapeToGPUPass(const NativeTapeToGPUPass &other) : PassWrapper(other) {}
  Option<std::string> backend{*this, "backend", llvm::cl::desc("Private allocation lowering: nvidia or rocm"), llvm::cl::init("nvidia")};
  llvm::StringRef getArgument() const final { return "tessera-native-tape-to-gpu"; }
  llvm::StringRef getDescription() const final { return "Materialize a bounded static bufferized AD product as a serial GPU entry"; }
  void getDependentDialects(mlir::DialectRegistry &r) const override {
    r.insert<mlir::arith::ArithDialect,mlir::func::FuncDialect,mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,mlir::memref::MemRefDialect,mlir::scf::SCFDialect,mlir::math::MathDialect>();
  }
  void runOnOperation() override {
    using namespace mlir;
    auto m=getOperation();
    auto reject=[&]() { m.emitError("native tape GPU requires an isolated static f32 buffer product with bounded for/if control and at most 4096 temporary bytes"); signalPassFailure(); };
    if (backend!="nvidia" && backend!="rocm") return reject();
    SmallVector<func::FuncOp> fs(m.getOps<func::FuncOp>());
    if (fs.size()!=1 || !m->hasAttr("tessera.autodiff.product_abi") || !m->hasAttr("tessera.autodiff.product_pair")) return reject();
    auto f=fs[0];
    if (!f.getBody().hasOneBlock() || f.getNumResults() || !isa<func::ReturnOp>(f.getBody().front().getTerminator())) return reject();
    auto admissible=[](Type t) {
      auto mt=dyn_cast<MemRefType>(t);
      if (!mt || !mt.hasStaticShape() || !mt.getElementType().isF32() || !mt.getLayout().isIdentity() || mt.getMemorySpace()) return false;
      int64_t count=1;
      for (auto d:mt.getShape()) {
        if (d<=0 || d>1024 || count>1024/d) return false;
        count*=d;
      }
      return true;
    };
    for (auto t:f.getArgumentTypes()) if (!admissible(t)) return reject();
    bool bad=false;
    int64_t bytes=0;
    f.walk([&](Operation *op) {
      auto dialect=op->getName().getDialectNamespace();
      if (op!=f.getOperation() && dialect!="arith" && dialect!="math" && dialect!="memref" && dialect!="scf" && !isa<func::ReturnOp>(op)) bad=true;
      if (isa<scf::WhileOp,func::CallOp,memref::DeallocOp>(op)) bad=true;
      if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        auto source=cast<MemRefType>(copy.getSource().getType());
        auto target=cast<MemRefType>(copy.getTarget().getType());
        // The serial copy expansion uses compile-time bounds, never the
        // dynamic-dimension sentinel as a loop extent.
        if (!source.hasStaticShape() || !target.hasStaticShape() || source.getShape()!=target.getShape()) bad=true;
      }
      if (op->getNumRegions() && op!=f.getOperation() && !isa<scf::ForOp,scf::IfOp>(op)) bad=true;
      if (auto loop=dyn_cast<scf::ForOp>(op)) {
        APInt lo,hi,step;
        if (!matchPattern(loop.getLowerBound(),m_ConstantInt(&lo)) || !matchPattern(loop.getUpperBound(),m_ConstantInt(&hi)) ||
            !matchPattern(loop.getStep(),m_ConstantInt(&step)) || step.getSExtValue()<=0 || step.getSExtValue()>1024 || lo.getSExtValue()<0 || lo.getSExtValue()>1024 || hi.getSExtValue()<0 || hi.getSExtValue()>1024) bad=true;
      }
      if (!isa<memref::AllocOp,memref::AllocaOp,memref::GetGlobalOp>(op)) return;
      if (!admissible(op->getResult(0).getType())) { bad=true; return; }
      int64_t n=cast<MemRefType>(op->getResult(0).getType()).getNumElements()*4;
      for (Operation *parent=op->getParentOp();parent && parent!=f.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) {
          APInt lo,hi,step;
          if (!matchPattern(loop.getLowerBound(),m_ConstantInt(&lo)) || !matchPattern(loop.getUpperBound(),m_ConstantInt(&hi)) || !matchPattern(loop.getStep(),m_ConstantInt(&step)) || step.getSExtValue()<=0 || step.getSExtValue()>1024 || lo.getSExtValue()<0 || lo.getSExtValue()>1024 || hi.getSExtValue()<0 || hi.getSExtValue()>1024) { bad=true; return; }
          int64_t trip=std::max<int64_t>(1,(hi.getSExtValue()-lo.getSExtValue()+step.getSExtValue()-1)/step.getSExtValue());
          if (trip>1024 || (trip && n>4096/trip)) { bad=true; return; }
          n*=trip;
        }
      bytes+=n;
      if (bytes>4096) bad=true;
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto value=global ? dyn_cast_or_null<DenseFPElementsAttr>(global.getInitialValueAttr()) : DenseFPElementsAttr();
        if (!global || !global.getConstant() || !value || !value.isSplat()) bad=true;
      }
    });
    if (bad) return reject();
    std::string text; llvm::raw_string_ostream os(text);
    os<<"module { gpu.module @native_tape { gpu.func @product(";
    for (unsigned i=0;i<f.getNumArguments();++i) os<<"%p"<<i<<": !llvm.ptr<1>, ";
    os<<"%scratch: index) kernel {\n";
    for (auto [i,t]:llvm::enumerate(f.getArgumentTypes())) {
      auto mt=cast<MemRefType>(t); auto rank=mt.getRank();
      std::string desc; llvm::raw_string_ostream ds(desc);
      ds<<"!llvm.struct<(ptr, ptr, i64, array<"<<rank<<" x i64>, array<"<<rank<<" x i64>)>"; ds.flush();
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
    SmallVector<Operation *> memory;
    kernel.walk([&](Operation *op){if (isa<memref::AllocOp,memref::GetGlobalOp,memref::CopyOp>(op) || (isa<memref::AllocaOp>(op) && cast<MemRefType>(op->getResult(0).getType()).hasStaticShape())) memory.push_back(op);});
    for (auto *op:memory) {
      OpBuilder at(op); auto loc=op->getLoc();
      auto loopNest=[&](MemRefType type,auto leaf) {
        SmallVector<Value> indices;
        std::function<void(int)> recurse=[&](int dim) {
          if (dim==type.getRank()) { leaf(at,indices); return; }
          auto zero=arith::ConstantIndexOp::create(at,loc,0), one=arith::ConstantIndexOp::create(at,loc,1), end=arith::ConstantIndexOp::create(at,loc,type.getDimSize(dim));
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
      // A GPU static alloca is not a fresh allocation on each loop trip.
      // Give every syntactic allocation an entry-owned byte array and derive
      // a distinct slot from the complete enclosing iteration path.
      SmallVector<scf::ForOp> loops;
      int64_t slots=1;
      for (auto *parent=op->getParentOp();parent!=kernel.getOperation();parent=parent->getParentOp())
        if (auto loop=dyn_cast<scf::ForOp>(parent)) loops.push_back(loop);
      Value slot=arith::ConstantIndexOp::create(at,loc,0);
      for (auto loop:llvm::reverse(loops)) {
        APInt lo,hi,step;
        matchPattern(loop.getLowerBound(),m_ConstantInt(&lo));
        matchPattern(loop.getUpperBound(),m_ConstantInt(&hi));
        matchPattern(loop.getStep(),m_ConstantInt(&step));
        int64_t trip=std::max<int64_t>(1,(hi.getSExtValue()-lo.getSExtValue()+step.getSExtValue()-1)/step.getSExtValue());
        slots*=trip;
        auto count=arith::ConstantIndexOp::create(at,loc,trip);
        auto delta=arith::SubIOp::create(at,loc,loop.getInductionVar(),loop.getLowerBound());
        auto ordinal=arith::DivUIOp::create(at,loc,delta,loop.getStep());
        slot=arith::AddIOp::create(at,loc,arith::MulIOp::create(at,loc,slot,count),ordinal);
      }
      int64_t slotBytes=type.getNumElements()*4;
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
      auto view=memref::ViewOp::create(at,loc,privateType,root,offset,ValueRange{});
      Value local=view;
      if (space) local=memref::MemorySpaceCastOp::create(at,loc,type,view);
      if (auto get=dyn_cast<memref::GetGlobalOp>(op)) {
        auto global=m.lookupSymbol<memref::GlobalOp>(get.getName());
        auto value=cast<DenseFPElementsAttr>(global.getInitialValueAttr()).getSplatValue<APFloat>();
        loopNest(type,[&](OpBuilder &builder,ValueRange indices) {
          auto v=arith::ConstantOp::create(builder,loc,builder.getF32Type(),builder.getFloatAttr(builder.getF32Type(),value));
          memref::StoreOp::create(builder,loc,v,local,indices);
        });
      }
      op->getResult(0).replaceAllUsesWith(local); op->erase();
    }
    m.getBody()->clear();
    m.getBody()->getOperations().splice(m.getBody()->end(),gpu->getBody()->getOperations());
    m->setAttr("tessera.autodiff.temporary_bytes",IntegerAttr::get(IntegerType::get(m.getContext(),64),bytes));
  }
};
}
std::unique_ptr<mlir::Pass> createNativeTapeToGPUPass() { return std::make_unique<NativeTapeToGPUPass>(); }
}
