// Native source-preserving NVFP4 expansion. One thread owns one K16 group.
// The checked launch boundary validates finite nonnegative E4M3 scales,
// positive finite projection globals, and representable rounded f32 products.
// A runtime-K image takes the K16 group count as an i64 scalar and launches
// a (groups/256, N) grid: grid y is the row, so no lane divides by the
// runtime extent. Every address is the static kernel's at that K.
// A runtime-N image keeps the flat grid and static K16 group count, so its
// row/group split divides by a constant; N is one i64 scalar. Its P projection
// boundaries are equally spaced and derived at launch as p*(N/P).
static mlir::LogicalResult emitNativeNVFP4Expand(mlir::ModuleOp module,
                                                mlir::Operation *op) {
  using namespace mlir;
  auto arch = module->getAttrOfType<StringAttr>("tessera.arch");
  if (!arch || arch.getValue() != "gfx1201")
    return op->emitError("native NVFP4 expansion requires an exact gfx1201 module");
  int64_t n = op->getAttrOfType<IntegerAttr>("n").getInt();
  int64_t k = op->getAttrOfType<IntegerAttr>("k").getInt();
  const bool runtimeK = op->hasAttr("runtime_k");
  const bool runtimeN = op->hasAttr("runtime_n");
  if (runtimeK != (k == 0) || runtimeN != (n == 0) || (runtimeK && runtimeN) ||
      (runtimeN && op->getAttrOfType<ArrayAttr>("row_offsets").size() < 2))
    return op->emitError("NVFP4 expansion extents must be static or one runtime image identity");
  int64_t groups = k / 16;
  auto offsets = op->getAttrOfType<ArrayAttr>("row_offsets");
  std::string name = op->getAttrOfType<StringAttr>("name").getValue().str();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location l = op->getLoc();
  auto gm = b.create<gpu::GPUModuleOp>(l,name + "_mod");
  b.setInsertionPointToStart(&gm.getBodyRegion().front());
  auto bytes = MemRefType::get({ShapedType::kDynamic},b.getI8Type());
  auto doubles = MemRefType::get({ShapedType::kDynamic},b.getF64Type());
  auto floats = MemRefType::get({ShapedType::kDynamic},b.getF32Type());
  SmallVector<Type> arguments{bytes,bytes,doubles,bytes,floats};
  if (runtimeK || runtimeN) arguments.push_back(b.getI64Type());
  auto f = b.create<gpu::GPUFuncOp>(l,name,b.getFunctionType(arguments,{}));
  f.setKernelAttr(b.getUnitAttr());
  f->setAttr("gpu.known_block_size",b.getDenseI32ArrayAttr({256,1,1}));
  b.setInsertionPointToStart(&f.getBody().front());
  auto ci = [&](int64_t v) -> Value { return b.create<arith::ConstantIndexOp>(l,v); };
  auto c32 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l,v,32); };
  auto c64 = [&](int64_t v) -> Value { return b.create<arith::ConstantIntOp>(l,v,64); };
  auto choose = [&](Value p,Value a,Value v) -> Value {
    return b.create<arith::SelectOp>(l,p,a,v);
  };
  auto eq = [&](Value a,Value v) -> Value {
    return b.create<arith::CmpIOp>(l,arith::CmpIPredicate::eq,a,v);
  };
  Value block = b.create<gpu::BlockIdOp>(l,gpu::Dimension::x);
  Value thread = b.create<gpu::ThreadIdOp>(l,gpu::Dimension::x);
  Value lane = b.create<arith::AddIOp>(
      l,b.create<arith::MulIOp>(l,block,ci(256)),thread);
  Value index, row, group, active;
  if (runtimeK) {
    Value count = b.create<arith::IndexCastOp>(
        l,b.getIndexType(),f.getArgument(5));
    row = b.create<gpu::BlockIdOp>(l,gpu::Dimension::y);
    group = lane;
    active = b.create<arith::AndIOp>(l,
        b.create<arith::CmpIOp>(l,arith::CmpIPredicate::ult,group,count),
        b.create<arith::CmpIOp>(l,arith::CmpIPredicate::ult,row,ci(n)));
  } else if (runtimeN) {
    index = lane;
    Value rows = b.create<arith::IndexCastOp>(l,b.getIndexType(),f.getArgument(5));
    active = b.create<arith::CmpIOp>(l,arith::CmpIPredicate::ult,index,
        b.create<arith::MulIOp>(l,rows,ci(groups)));
  } else {
    index = lane;
    active = b.create<arith::CmpIOp>(
        l,arith::CmpIPredicate::ult,index,ci(n*groups));
  }
  auto guard = b.create<scf::IfOp>(l,active,false);
  b.setInsertionPointToStart(guard.thenBlock());
  if (runtimeK) {
    Value count = b.create<arith::IndexCastOp>(
        l,b.getIndexType(),f.getArgument(5));
    index = b.create<arith::AddIOp>(
        l,b.create<arith::MulIOp>(l,row,count),group);
  } else {
    row = b.create<arith::DivUIOp>(l,index,ci(groups));
    group = b.create<arith::RemUIOp>(l,index,ci(groups));
  }
  Value projection = ci(0);
  // The verifier proved a runtime-N image's boundaries equally spaced.
  Value width = runtimeN && offsets.size() > 2
      ? Value(b.create<arith::DivUIOp>(l,
            b.create<arith::IndexCastOp>(l,b.getIndexType(),f.getArgument(5)),
            ci(int64_t(offsets.size()) - 1))) : Value();
  for (unsigned p=1; p+1<offsets.size(); ++p) {
    Value boundary = runtimeN
        ? Value(b.create<arith::MulIOp>(l,width,ci(p)))
        : ci(cast<IntegerAttr>(offsets[p]).getInt());
    Value after = b.create<arith::CmpIOp>(
        l,arith::CmpIPredicate::uge,row,boundary);
    projection = choose(after,ci(p),projection);
  }
  Value global = b.create<memref::LoadOp>(l,f.getArgument(2),ValueRange{projection});
  Value raw = b.create<memref::LoadOp>(l,f.getArgument(1),ValueRange{index});
  Value bits = b.create<arith::ExtUIOp>(l,b.getI32Type(),raw);
  Value exponent = b.create<arith::AndIOp>(
      l,b.create<arith::ShRUIOp>(l,bits,c32(3)),c32(15));
  Value fraction = b.create<arith::AndIOp>(l,bits,c32(7));
  Value wideExp = b.create<arith::ExtUIOp>(l,b.getI64Type(),exponent);
  Value powBits = b.create<arith::ShLIOp>(
      l,b.create<arith::AddIOp>(l,wideExp,c64(1013)),c64(52));
  Value pow = b.create<arith::BitcastOp>(l,b.getF64Type(),powBits);
  Value normal = b.create<arith::MulFOp>(
      l,b.create<arith::UIToFPOp>(l,b.getF64Type(),
        b.create<arith::AddIOp>(l,fraction,c32(8))),pow);
  Value subPow = b.create<arith::ConstantOp>(l,b.getF64Type(),b.getF64FloatAttr(0x1p-9));
  Value subnormal = b.create<arith::MulFOp>(
      l,b.create<arith::UIToFPOp>(l,b.getF64Type(),fraction),subPow);
  Value magnitude = choose(eq(exponent,c32(0)),subnormal,normal);
  Value negative = b.create<arith::CmpIOp>(
      l,arith::CmpIPredicate::ne,b.create<arith::AndIOp>(l,bits,c32(128)),c32(0));
  Value signedScale = choose(negative,b.create<arith::NegFOp>(l,magnitude),magnitude);
  Value product = b.create<arith::MulFOp>(l,signedScale,global);
  Value rounded = b.create<arith::TruncFOp>(l,b.getF32Type(),product);
  Value columns = runtimeN
      ? Value(b.create<arith::IndexCastOp>(l,b.getIndexType(),f.getArgument(5))) : ci(n);
  Value scaleIndex = b.create<arith::AddIOp>(
      l,b.create<arith::MulIOp>(l,group,columns),row);
  b.create<memref::StoreOp>(l,rounded,f.getArgument(4),ValueRange{scaleIndex});
  Value packedBase = b.create<arith::MulIOp>(l,index,ci(8));
  Value valueBase = b.create<arith::MulIOp>(l,index,ci(16));
  for (int j=0; j<8; ++j) {
    Value address = b.create<arith::AddIOp>(l,packedBase,ci(j));
    Value packed = b.create<arith::ExtUIOp>(
        l,b.getI32Type(),b.create<memref::LoadOp>(l,f.getArgument(0),ValueRange{address}));
    for (int half=0; half<2; ++half) {
      Value code = b.create<arith::AndIOp>(
          l,b.create<arith::ShRUIOp>(l,packed,c32(4*half)),c32(15));
      Value mag = b.create<arith::AndIOp>(l,code,c32(7));
      Value sign = b.create<arith::ShLIOp>(
          l,b.create<arith::AndIOp>(l,code,c32(8)),c32(4));
      Value expanded = b.create<arith::ShLIOp>(
          l,b.create<arith::AddIOp>(l,mag,c32(12)),c32(2));
      expanded = choose(eq(mag,c32(1)),c32(48),expanded);
      expanded = choose(eq(mag,c32(0)),c32(0),expanded);
      Value out = b.create<arith::TruncIOp>(
          l,b.getI8Type(),b.create<arith::OrIOp>(l,expanded,sign));
      Value destination = b.create<arith::AddIOp>(l,valueBase,ci(j*2+half));
      b.create<memref::StoreOp>(l,out,f.getArgument(3),ValueRange{destination});
    }
  }
  b.setInsertionPointAfter(guard);
  b.create<gpu::ReturnOp>(l);
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),b.getUnitAttr());
  return success();
}
