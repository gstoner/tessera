"""Independent streaming Q/K/V attention JVP oracle; no native execution claim.

Carry normalization, value and directional-score moments using the same max
rescaling. The max is a numerical gauge: its derivative cancels from the final
normalized result. Persistent accumulator storage is O(Dv), with bounded input tiles; no
quadratic score tensor is materialized.
"""
import numpy as np


def streaming_row_jvp(q, k, v, dq, dk, dv, *, scale, mask, block_size=8):
    q,k,v,dq,dk,dv = (np.asarray(x,dtype=np.float64) for x in (q,k,v,dq,dk,dv))
    mask=np.asarray(mask)
    if (q.ndim!=1 or k.ndim!=2 or v.ndim!=2 or k.shape[0]!=v.shape[0] or
            k.shape[1]!=q.size or dq.shape!=q.shape or dk.shape!=k.shape or dv.shape!=v.shape or
            mask.shape!=(k.shape[0],) or mask.dtype!=np.bool_ or type(block_size) is not int or block_size<1 or
            isinstance(scale,bool) or not np.isfinite(scale) or scale<=0):
        raise ValueError('invalid streaming attention row contract')
    maximum=-np.inf
    mass=0.
    value=np.zeros(v.shape[1])
    score_moment=0.
    directional_value=np.zeros(v.shape[1])
    for start in range(0,len(k),block_size):
        keep=mask[start:start+block_size]
        if not keep.any():
            continue
        kb,vb,dkb,dvb=(x[start:start+block_size][keep] for x in (k,v,dk,dv))
        scores=scale*(kb@q)
        directions=scale*(kb@dq+dkb@q)
        next_max=max(maximum,float(scores.max()))
        rescale=np.exp(maximum-next_max)
        weights=np.exp(scores-next_max)
        mass=mass*rescale+weights.sum()
        value=value*rescale+weights@vb
        score_moment=score_moment*rescale+weights@directions
        directional_value=directional_value*rescale+(weights*directions)@vb+weights@dvb
        maximum=next_max
    if mass==0:
        return value,directional_value,-np.inf
    primal=value/mass
    tangent=(directional_value-primal*score_moment)/mass
    return primal,tangent,maximum+np.log(mass)
