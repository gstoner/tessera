"""Typed compiler exports for split AD products, including nested residual IR.

These are native IR artifacts, not executable device allocations or proof that
arbitrary tensor tapes have been lowered to a backend's storage ABI.
"""
from dataclasses import dataclass
import hashlib
import json
import re
from .native_gpu_storage import _decode_image
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class NativeTapeProducts:
    forward_ir: str
    backward_ir: str
    paired_ir: str
    residual_types: tuple[str,...]
    residual_sources: tuple[str,...]
    digest: str


def export_native_tape_products(source):
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError('typed tape export requires production tessera-opt')
    products=[]
    contracts=[]
    lineages=[]
    for role in ('forward','backward'):
        text=run_tessera_opt(tool,source,'--tessera-autodiff-paired=export-product='+role)
        def attr(name):
            matches=re.findall(r'tessera\.autodiff\.'+name+r' = "((?:\\.|[^"\\])*)"',text)
            if len(matches)!=1:
                raise ValueError('native tape product lacks unique compiler ABI/lineage')
            return _decode_image(matches[0]).decode()
        contract=json.loads(attr('product_abi'))
        if contract.get('schema')!=1 or contract.get('role')!=role:
            raise ValueError('native tape product schema or role disagrees')
        products.append(text)
        contracts.append(contract)
        lineages.append(attr('product_pair'))
    forward,backward=contracts
    count=forward['primal_results']
    residuals=tuple(forward['results'][count:])
    offset=backward['primal_inputs']+backward['primal_results']
    if (lineages[0]!=lineages[1] or forward['residual_sources']!=backward['residual_sources'] or
            residuals!=tuple(backward['inputs'][offset:]) or len(residuals)!=len(forward['residual_sources'])):
        raise ValueError('native tape producer/consumer residual identity disagrees')
    digest=hashlib.sha256(json.dumps(products,separators=(',',':')).encode()).hexdigest()
    return NativeTapeProducts(products[0],products[1],lineages[0],residuals,tuple(forward['residual_sources']),digest)
