"""Independent math and store-boundary certificate for quantized MLP evidence.

This module is a benchmark/test oracle, never a production execution route.
"""
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import reference_nvfp4_expand,nvfp4_expansion_policy
from benchmarks.rocm.benchmark_public_quantized_gate_up import oracle as gate_oracle


def _parent_product(package,q,b,sa,sb,shape):
    output=np.empty((shape.m,shape.n),dtype=ml_dtypes.bfloat16)
    artifact=rt.RuntimeArtifact(metadata={"target":package.image.target},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    result=rt.launch(artifact,{"buffers":{"a":q,"b":b,"a_scale":sa,"b_scale":sb,"o":output},
        "scalars":{"M":shape.m,"N":shape.n,"K":shape.k}})
    if not result.get("ok") or result.get("execution_kind")!="native_gpu":
        raise AssertionError(result)
    return output


def down_reference(roots,activation):
    m,h=activation.shape;n=roots[4].shape[0]
    blocks=activation.astype(np.float32).reshape(m,h//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(*roots[4:],row_offsets=[0,n],numeric_policy=nvfp4_expansion_policy())
    quantized=q*sa.astype(np.float64)[...,None]
    weights=b.astype(np.float64).reshape(n,h//16,16)*sb.astype(np.float64).T[...,None]
    return np.einsum('mgj,ngj->mn',quantized,weights),quantized,weights


def oracle(roots):
    output,_,_=down_reference(roots,gate_oracle(roots[:4]))
    return output.astype(np.float32).astype(ml_dtypes.bfloat16)


def assert_compiled_chain(package,roots,actual,lib):
    # Diagnostic staged owners prove resident wiring independently. Production
    # execution above is one seven-step owner and has no intermediate readback.
    with package.gate.prepare(roots[:4],runtime_library=lib) as owner:
        generation,_=owner.invoke();activation=owner.read(generation)[0]
    expected_activation=gate_oracle(roots[:4])
    # Validate at the explicit parent store: a single BF16 midpoint crossing
    # can be amplified by SiLU even when its implementation is exact.
    x,packed,weight_scales,globals_=roots[:4];m,k=x.shape;h=activation.shape[1]
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).reshape(m,k)
    b,sb=reference_nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,h,2*h],numeric_policy=nvfp4_expansion_policy())
    parent_reference=np.einsum('mgj,ngj,mg,gn->mn',q.astype(np.float64).reshape(m,k//16,16),
        b.astype(np.float64).reshape(2*h,k//16,16),sa.astype(np.float64),sb.astype(np.float64))
    parent=_parent_product(package.gate.product,q,b,sa,sb,package.gate.product_recipe.shape).astype(np.float32)
    parent_stored=parent_reference.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(parent,parent_stored,rtol=.015625,atol=.00002)
    gate,up=parent[:,:h],parent[:,h:]
    activation_at_parent=(gate/(np.float32(1)+np.exp(-gate))*up).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(activation.astype(np.float32),activation_at_parent.astype(np.float32),rtol=.015625,atol=.00002)
    with package.down.prepare([activation,*roots[4:]],runtime_library=lib) as owner:
        generation,_=owner.invoke();staged=owner.read(generation)[0]
    np.testing.assert_array_equal(actual,staged)
    live,q_live,weights=down_reference(roots,activation)
    reference,q_reference,_=down_reference(roots,expected_activation)
    live_stored=live.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    reference_stored=reference.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    # The existing per-stage gate is unchanged. Quantization is discontinuous:
    # carrying a tolerated BF16 perturbation through a second quantizer cannot
    # reuse the same pointwise relative gate against the ideal earlier value.
    np.testing.assert_allclose(actual.astype(np.float32),live_stored,rtol=.015625,atol=.00002)
    propagation=np.einsum('mgj,ngj->mn',np.abs(q_live-q_reference),np.abs(weights))
    store_rounding=np.abs(live_stored-live)+np.abs(reference_stored-reference)
    numerical=.015625*np.abs(live_stored)+.00002
    assert np.all(np.abs(actual.astype(np.float64)-reference_stored)<=propagation+store_rounding+numerical)

    return {"resident_staged_bitwise_equal":True,"parent_store_max_abs_error":float(np.max(np.abs(parent-parent_stored))),
        "activation_at_parent_max_abs_error":float(np.max(np.abs(activation.astype(np.float32)-activation_at_parent.astype(np.float32)))),
        "down_at_activation_max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-live_stored))),
        "full_independent_max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-reference_stored))),
        "full_independent_outside_flat_gate":int(np.count_nonzero(np.abs(actual.astype(np.float64)-reference_stored)>(.015625*np.abs(reference_stored)+.00002))),
        "propagated_bound_passed":True,"propagation_max_abs_bound":float(propagation.max()),
        "per_stage_rtol":.015625,"per_stage_atol":.00002}

