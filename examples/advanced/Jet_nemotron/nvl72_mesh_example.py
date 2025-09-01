
from tessera import dist
mesh = dist.mesh(devs=72, topology="nvl72")
layout = {"batch":"dp", "seq":"tp", "params":"rp"}
