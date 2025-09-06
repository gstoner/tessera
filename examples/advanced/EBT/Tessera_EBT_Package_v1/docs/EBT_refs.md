# EBT references (curated)
- Project page: https://energy-based-transformers.github.io/
- Paper: https://arxiv.org/abs/2507.02092 (PDF linked on page)
- Code: https://github.com/alexiglad/EBT

Key claims summarized (see main design doc for details):
- EBT reframes prediction as *energy minimization* over candidate outputs.
- “System‑2 thinking”: do extra inner‑loop compute to refine predictions; optionally *self‑verify* by generating n candidates and selecting minimum energy.
- Reported scaling advantages vs feed‑forward Transformers (data/params/depth/FLOPs) and larger gains from extra compute at inference.
