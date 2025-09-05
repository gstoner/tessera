include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "CollectiveTypes.td"
class TesseraCollectiveOp<string m, list<OpTrait> T = []> : Op<Tessera_Collective_Dialect, m, T>;
def Attr_WireDType: StrEnumAttr<"WireDType","wire dtype",["fp32","bf16","fp16","fp8","i8"]>;
def AllToAllOp : TesseraCollectiveOp<"all_to_all", [MemoryEffects<[MemWrite, MemRead]>]> {
  let arguments = (ins AnyMemRef:$input, OptionalAttr<Attr_WireDType>:$dtype, OptionalAttr<I64Attr>:$chunk_bytes);
  let results = (outs Tessera_FutureType:$result);
  let assemblyFormat = [{ $input attr-dict }];
}
def PackCastOp : Op<Tessera_Collective_Dialect, "pack_cast", []> {
  let arguments = (ins AnyMemRef:$input, OptionalAttr<Attr_WireDType>:$to_dtype);
  let results   = (outs AnyMemRef:$packed);
  let assemblyFormat = [{ $input attr-dict }];
}
def AwaitOp : Op<Tessera_Collective_Dialect, "await", []> {
  let arguments = (ins Tessera_FutureType:$future); let results = (outs AnyType:$value);
  let assemblyFormat = [{ $future attr-dict }];
}
def QoSLimitOp   : Op<Tessera_Collective_Dialect, "qos.limit",   []> { let arguments = (ins I64Attr:$max_inflight); let results = (outs); let assemblyFormat = [{ attr-dict }]; }
def QoSAcquireOp : Op<Tessera_Collective_Dialect, "qos.acquire", []> { let arguments = (ins); let results = (outs); let assemblyFormat = [{ attr-dict }]; }
def QoSReleaseOp : Op<Tessera_Collective_Dialect, "qos.release", []> { let arguments = (ins); let results = (outs); let assemblyFormat = [{ attr-dict }]; }
