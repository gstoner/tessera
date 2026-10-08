// Lossless storage bridge for ROCM-NVFP4-INGEST-1. No FP arithmetic.
// Each source byte has one unique fragment-order destination; scales are
// copied group-major and one thread per row reduces the unsigned row maximum.
static mlir::LogicalResult emitNativeMXFP4Storage(mlir::ModuleOp module,
                                                 mlir::Operation *op) {
  using namespace mlir;
  int64_t n = op->getAttrOfType<IntegerAttr>("n").getInt();
  int64_t k = op->getAttrOfType<IntegerAttr>("k").getInt();
  std::string name = op->getAttrOfType<StringAttr>("name").getValue().str();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location l = op->getLoc();
  auto gm = b.create<gpu::GPUModuleOp>(l,name+"_mod");
  b.setInsertionPointToStart(&gm.getBodyRegion().front());
  auto bytes = MemRefType::get({ShapedType::kDynamic},b.getI8Type());
  auto fn = b.create<gpu::GPUFuncOp>(l,name,
      b.getFunctionType({bytes,bytes,bytes,bytes},{}));
  fn.setKernelAttr(b.getUnitAttr());
  fn->setAttr("gpu.known_block_size",b.getDenseI32ArrayAttr({256,1,1}));
  b.setInsertionPointToStart(&fn.getBody().front());
  auto ci = [&](int64_t v) -> Value { return b.create<arith::ConstantIndexOp>(l,v); };
  auto add = [&](Value a,Value v) -> Value { return b.create<arith::AddIOp>(l,a,v); };
  auto mul = [&](Value a,Value v) -> Value { return b.create<arith::MulIOp>(l,a,v); };
  auto div = [&](Value a,Value v) -> Value { return b.create<arith::DivUIOp>(l,a,v); };
  auto rem = [&](Value a,Value v) -> Value { return b.create<arith::RemUIOp>(l,a,v); };
  Value index=add(mul(b.create<gpu::BlockIdOp>(l,gpu::Dimension::x),ci(256)),
                  b.create<gpu::ThreadIdOp>(l,gpu::Dimension::x));
  auto within = [&](int64_t extent) -> Value {
    return b.create<arith::CmpIOp>(l,arith::CmpIPredicate::ult,index,ci(extent));
  };
  auto codeGuard=b.create<scf::IfOp>(l,within(n*(k/2)),false);
  b.setInsertionPointToStart(codeGuard.thenBlock());
  Value row=div(index,ci(k/2)), byte=rem(index,ci(k/2));
  // [N/16,K/16,2,16,4] bytes, retaining adjacent low/high K nibbles.
  Value tile=add(mul(div(row,ci(16)),ci(k/16)),div(byte,ci(8)));
  Value destination=add(add(mul(tile,ci(128)),mul(div(rem(byte,ci(8)),ci(4)),ci(64))),
                        add(mul(rem(row,ci(16)),ci(4)),rem(byte,ci(4))));
  Value value=b.create<memref::LoadOp>(l,fn.getArgument(0),ValueRange{index});
  b.create<memref::StoreOp>(l,value,fn.getArgument(2),ValueRange{destination});
  b.setInsertionPointAfter(codeGuard);
  auto scaleGuard=b.create<scf::IfOp>(l,within(n*(k/32)),false);
  b.setInsertionPointToStart(scaleGuard.thenBlock());
  Value scale=b.create<memref::LoadOp>(l,fn.getArgument(1),ValueRange{index});
  b.create<memref::StoreOp>(l,scale,fn.getArgument(3),ValueRange{index});
  b.setInsertionPointAfter(scaleGuard);
  auto rowGuard=b.create<scf::IfOp>(l,within(n),false);
  b.setInsertionPointToStart(rowGuard.thenBlock());
  Value zero=b.create<arith::ConstantIntOp>(l,0,8);
  auto maximum=b.create<scf::ForOp>(l,ci(0),ci(k/32),ci(1),ValueRange{zero},
      [&](OpBuilder &body,Location loc,Value group,ValueRange carried) {
        Value address=body.create<arith::AddIOp>(loc,
            body.create<arith::MulIOp>(loc,group,ci(n)),index);
        Value raw=body.create<memref::LoadOp>(loc,fn.getArgument(1),ValueRange{address});
        Value result=body.create<arith::MaxUIOp>(loc,carried[0],raw);
        body.create<scf::YieldOp>(loc,ValueRange{result});
      });
  b.create<memref::StoreOp>(l,maximum.getResult(0),fn.getArgument(3),
      ValueRange{add(ci(n*(k/32)),index)});
  b.setInsertionPointToEnd(&fn.getBody().front());
  b.create<gpu::ReturnOp>(l);
  return success();
}
