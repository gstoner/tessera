//===- ScheduleMeshPipelineOps.td - Mesh & pipeline ops ------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ScheduleDialect : Dialect {
  let name = "schedule";
  let cppNamespace = "::tessera::schedule";
}

def Schedule_MeshDefineOp : Op<ScheduleDialect, "mesh.define"> {
  let summary = "Define a logical mesh";
  let arguments = (ins ArrayAttr:$dims, ArrayAttr:$axis_names);
  let hasVerifier = 1;
}

def Schedule_MeshRegionOp : Op<ScheduleDialect, "mesh.region"> {
  let summary = "Enter a mesh axis region";
  let arguments = (ins SymbolRefAttr:$mesh, StrAttr:$axis);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_PipelineRegionOp : Op<ScheduleDialect, "pipeline.region"> {
  let summary = "Pipeline-parallel region with schedule";
  let arguments = (ins StrAttr:$schedule, I32Attr:$micro_batches);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_StageOp : Op<ScheduleDialect, "stage"> {
  let summary = "A pipeline stage";
  let arguments = (ins ArrayAttr:$devices);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_YieldOp : Op<ScheduleDialect, "yield",
    [Pure, Terminator, ReturnLike]> {
  let summary = "Terminator for schedule region bodies";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [OpBuilder<(ins), [{
    $_state.addTypes({});
  }]>];
}

def Schedule_TileOp : Op<ScheduleDialect, "tile"> {
  let summary = "Materialized graph-op tile plan";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       OptionalAttr<I64Attr>:$tile_m,
                       OptionalAttr<I64Attr>:$tile_n,
                       OptionalAttr<I64Attr>:$tile_k,
                       OptionalAttr<I64Attr>:$tile_h,
                       OptionalAttr<I64Attr>:$tile_w,
                       OptionalAttr<I64Attr>:$tile_c);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

// Mixed-level E2E spine contract. The Graph op remains the semantic producer
// while this SSA edge carries the selected schedule to Schedule->Tile. The
// lowering must consume both this op and its defining Graph matmul atomically.
def Schedule_MatmulOp : Op<ScheduleDialect, "matmul", [Pure]> {
  let summary = "Scheduled Graph matmul value with content-addressed decisions";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       I64Attr:$tile_m,
                       I64Attr:$tile_n,
                       I64Attr:$tile_k,
                       I64Attr:$macro_tile_m,
                       I64Attr:$macro_tile_n,
                       I64Attr:$warps,
                       I64Attr:$pipeline_depth,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       StrAttr:$a_layout,
                       StrAttr:$b_layout,
                       StrAttr:$raster_order,
                       I64Attr:$raster_group);
  let results = (outs AnyType:$scheduled);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($scheduled)";
  let hasVerifier = 1;
}

// Content-addressed semantic-kernel edges used by E2E-REAL-5.  These retain
// the Graph result as the subject until Schedule->Tile atomically replaces the
// pair with one launch-level physical kernel.
def Schedule_SoftmaxOp : Op<ScheduleDialect, "softmax", [Pure]> {
  let summary = "Scheduled Graph softmax with explicit numeric policy";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       I64Attr:$axis,
                       StrAttr:$exp_mode,
                       BoolAttr:$ftz,
                       I64Attr:$workgroup_size);
  let results = (outs AnyType:$scheduled);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($scheduled)";
  let hasVerifier = 1;
}

def Schedule_ReduceOp : Op<ScheduleDialect, "reduce", [Pure]> {
  let summary = "Scheduled Graph reduction with explicit numeric policy";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       StrAttr:$kind,
                       I64Attr:$axis,
                       BoolAttr:$keepdims,
                       StrAttr:$schedule,
                       StrAttr:$nan_mode,
                       BoolAttr:$inner_is_one,
                       I64Attr:$workgroup_size);
  let results = (outs AnyType:$scheduled);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($scheduled)";
  let hasVerifier = 1;
}

def Schedule_FFTOp : Op<ScheduleDialect, "fft", [Pure]> {
  let summary = "Content-addressed one-dimensional spectral transform plan";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       StrAttr:$mode,
                       I64Attr:$axis,
                       I64Attr:$length,
                       I64Attr:$batch,
                       BoolAttr:$inverse,
                       StrAttr:$normalization,
                       F32Attr:$scale,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       StrAttr:$radix_policy,
                       StrAttr:$strategy,
                       StrAttr:$algorithm,
                       DenseI64ArrayAttr:$radix_sequence,
                       I64Attr:$bluestein_m,
                       I64Attr:$workspace_elems,
                       StrAttr:$workspace_policy,
                       StrAttr:$residency,
                       StrAttr:$twiddle_layout,
                       StrAttr:$twiddle_policy,
                       BoolAttr:$deterministic,
                       StrAttr:$kernel_family,
                       I64Attr:$workgroup_size);
  let results = (outs AnyType:$scheduled);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($scheduled)";
  let hasVerifier = 1;
}

def Schedule_AttentionOp : Op<ScheduleDialect, "attention", [Pure]> {
  let summary = "Scheduled canonical rank-4 attention recurrence";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       F32Attr:$scale,
                       BoolAttr:$causal,
                       BoolAttr:$bias,
                       I64Attr:$window_left,
                       I64Attr:$window_right,
                       F32Attr:$softcap,
                       F32Attr:$dropout_p,
                       I64Attr:$dropout_seed,
                       I64Attr:$tile_q,
                       I64Attr:$tile_kv,
                       I64Attr:$workgroup_size,
                       StrAttr:$recurrence,
                       StrAttr:$backward_lse_policy,
                       StrAttr:$backward_lse_selection);
  let results = (outs AnyType:$scheduled);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($scheduled)";
  let hasVerifier = 1;
}

def Schedule_AttentionBackwardOp : Op<ScheduleDialect, "attention_backward", [Pure]> {
  let summary = "Scheduled canonical tensor-valued attention VJP";
  let arguments = (ins AnyType:$dq_subject,
                       AnyType:$dk_subject,
                       AnyType:$dv_subject,
                       StrAttr:$artifact_hash,
                       StrAttr:$arch,
                       StrAttr:$storage,
                       StrAttr:$accum,
                       F32Attr:$scale,
                       BoolAttr:$causal,
                       BoolAttr:$bias,
                       I64Attr:$window_left,
                       I64Attr:$window_right,
                       F32Attr:$softcap,
                       F32Attr:$dropout_p,
                       I64Attr:$dropout_seed,
                       I64Attr:$query_block,
                       I64Attr:$key_block,
                       I64Attr:$split_count,
                       DenseI64ArrayAttr:$reduction_order,
                       I64Attr:$workspace_bytes,
                       I64Attr:$workgroup_size,
                       StrAttr:$recurrence,
                       StrAttr:$lse_checkpoint_policy,
                       StrAttr:$lse_checkpoint_selection);
  let results = (outs AnyType:$dq_scheduled,
                      AnyType:$dk_scheduled,
                      AnyType:$dv_scheduled);
  let assemblyFormat = "$dq_subject `,` $dk_subject `,` $dv_subject attr-dict `:` type($dq_subject) `,` type($dk_subject) `,` type($dv_subject) `->` type($dq_scheduled) `,` type($dk_scheduled) `,` type($dv_scheduled)";
  let hasVerifier = 1;
}

def Schedule_LionVJPOp : Op<ScheduleDialect, "lion_vjp", [Pure]> {
  let summary = "Content-addressed Lion stop-sign VJP with buffer lineage";
  let arguments = (ins AnyType:$parameter,
                       AnyType:$gradient,
                       AnyType:$moment,
                       AnyType:$dparameter,
                       AnyType:$dmoment,
                       StrAttr:$artifact_hash,
                       StrAttr:$lineage_payload,
                       StrAttr:$arch,
                       F32Attr:$learning_rate,
                       F32Attr:$beta2,
                       F32Attr:$weight_decay,
                       StrAttr:$derivative_policy,
                       StrAttr:$mutation_mode,
                       StrAttr:$alias_policy,
                       StrAttr:$state_transition,
                       DenseI64ArrayAttr:$ordered_writes,
                       I64Attr:$workgroup_size);
  let results = (outs AnyType:$dparameter_scheduled,
                      AnyType:$dgradient_scheduled,
                      AnyType:$dmoment_scheduled);
  let assemblyFormat = "$parameter `,` $gradient `,` $moment `,` $dparameter `,` $dmoment attr-dict `:` type($parameter) `,` type($gradient) `,` type($moment) `,` type($dparameter) `,` type($dmoment) `->` type($dparameter_scheduled) `,` type($dgradient_scheduled) `,` type($dmoment_scheduled)";
  let hasVerifier = 1;
}

def Schedule_AdafactorVJPOp : Op<ScheduleDialect, "adafactor_vjp", [Pure]> {
  let summary = "Content-addressed full or factored Adafactor VJP";
  let arguments = (ins Variadic<AnyType>:$inputs,
                       StrAttr:$artifact_hash,
                       StrAttr:$lineage_payload,
                       StrAttr:$arch,
                       StrAttr:$topology,
                       F32Attr:$learning_rate,
                       F32Attr:$beta2,
                       F32Attr:$epsilon,
                       StrAttr:$mutation_mode,
                       StrAttr:$alias_policy,
                       StrAttr:$state_transition,
                       DenseI64ArrayAttr:$ordered_writes,
                       I64Attr:$workgroup_size);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($results)";
  let hasVerifier = 1;
}

def Schedule_SequenceMixerBackwardOp : Op<ScheduleDialect, "sequence_mixer_backward", [Pure]> {
  let summary = "Content-addressed checkpoint and reverse sequence-mixer program";
  let arguments = (ins AnyType:$q, AnyType:$k, AnyType:$v,
                       AnyType:$gate, AnyType:$beta, AnyType:$decay,
                       AnyType:$dy,
                       StrAttr:$artifact_hash,
                       StrAttr:$lineage_payload,
                       StrAttr:$arch,
                       StrAttr:$family,
                       BoolAttr:$erase,
                       I64Attr:$chunk_size,
                       BoolAttr:$parallel_chunks,
                       StrAttr:$mutation_mode,
                       StrAttr:$alias_policy,
                       StrAttr:$workspace_owner,
                       ArrayAttr:$phase_order,
                       I64Attr:$workgroup_size);
  let results = (outs AnyType:$dq, AnyType:$dk, AnyType:$dv,
                      AnyType:$dgate, AnyType:$dbeta, AnyType:$ddecay);
  let assemblyFormat = "$q `,` $k `,` $v `,` $gate `,` $beta `,` $decay `,` $dy attr-dict `:` type($q) `,` type($k) `,` type($v) `,` type($gate) `,` type($beta) `,` type($decay) `,` type($dy) `->` type($dq) `,` type($dk) `,` type($dv) `,` type($dgate) `,` type($dbeta) `,` type($ddecay)";
  let hasVerifier = 1;
}

def Schedule_ElementwiseOp : Op<ScheduleDialect, "elementwise"> {
  let summary = "Materialized elementwise or reduction schedule plan";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       I64Attr:$ordinal,
                       DefaultValuedAttr<BoolAttr, "true">:$vectorize);
  let assemblyFormat = "attr-dict";
}

def Schedule_LayoutOp : Op<ScheduleDialect, "layout"> {
  let summary = "Layout decision attached to scheduled operands";
  let arguments = (ins ArrayAttr:$operands,
                       StrAttr:$layout,
                       I64Attr:$ordinal);
  let assemblyFormat = "attr-dict";
}

def Schedule_WarpOp : Op<ScheduleDialect, "warp"> {
  let summary = "Warp role region for producer/consumer specialization";
  let arguments = (ins StrAttr:$role,
                       OptionalAttr<I32Attr>:$count);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_OptimizerShardOp : Op<ScheduleDialect, "optimizer_shard"> {
  let summary = "Optimizer-state sharding plan metadata";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$axis,
                       OptionalAttr<StrAttr>:$policy,
                       OptionalAttr<I64Attr>:$partitions);
  let results = (outs AnyType:$sharded);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($sharded)";
  let hasVerifier = 1;
}

def Schedule_PrefetchOp : Op<ScheduleDialect, "prefetch"> {
  let summary = "Prefetch a value/cache into an explicit memory space";
  let arguments = (ins AnyType:$source,
                       StrAttr:$into,
                       StrAttr:$overlap,
                       OptionalAttr<I32Attr>:$stage,
                       OptionalAttr<I32Attr>:$vector);
  let results = (outs AnyType:$staged);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($staged)";
  let hasVerifier = 1;
}

def Schedule_AsyncCopyOp : Op<ScheduleDialect, "async_copy"> {
  let summary = "Schedule-level asynchronous movement before Tile IR lowering";
  let arguments = (ins AnyType:$src, AnyType:$dst,
                       StrAttr:$src_space,
                       StrAttr:$dst_space,
                       I32Attr:$stage,
                       StrAttr:$overlap);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst) `->` type($token)";
  let hasVerifier = 1;
}

def Schedule_AwaitMovementOp : Op<ScheduleDialect, "await_movement"> {
  let summary = "Wait for a schedule-level movement token at the true use site";
  let arguments = (ins AnyType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
  let hasVerifier = 1;
}

def Schedule_ArtifactOp : Op<ScheduleDialect, "artifact"> {
  let summary = "Durable schedule artifact hash and selected tuning knobs";
  let arguments = (ins StrAttr:$hash,
                       StrAttr:$arch,
                       StrAttr:$shape_key,
                       OptionalAttr<DictionaryAttr>:$tile,
                       OptionalAttr<DictionaryAttr>:$movement,
                       OptionalAttr<StrAttr>:$numeric_policy);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Schedule_KnobOp : Op<ScheduleDialect, "knob"> {
  let summary = "Relaxed or discrete schedule-search knob for DNAS/autotuning";
  let arguments = (ins AnyType:$subject,
                       StrAttr:$name,
                       ArrayAttr:$choices,
                       OptionalAttr<ArrayAttr>:$logits,
                       DefaultValuedAttr<BoolAttr, "false">:$frozen);
  let results = (outs AnyType:$selected);
  let assemblyFormat = "$subject attr-dict `:` type($subject) `->` type($selected)";
  let hasVerifier = 1;
}
