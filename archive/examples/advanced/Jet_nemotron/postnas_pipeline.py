
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

@dataclass
class PostNASConfig:
    search_layers: List[int]
    keep_full_attn_budget: int
    jetblock_space: Dict[str, List]
    schedule_space: Dict[str, List]
    metric: str = "throughput_vs_mmlu"

class PostNASSearch:
    def __init__(self, model, cfg: PostNASConfig, eval_fn: Callable[[Any], Dict[str,float]]):
        self.model = model
        self.cfg = cfg
        self.eval_fn = eval_fn

    def freeze_mlps(self):
        for l in self.model.layers:
            l.mlp.requires_grad_(False)

    def candidate_architectures(self):
        layers = self.cfg.search_layers
        F = self.cfg.keep_full_attn_budget
        seeds = [set(layers[i::max(1,F)]) for i in range(max(1,F))]
        for seed in seeds:
            for f in self.cfg.jetblock_space.get('feature_map', ['elu1']):
                for ks in self.cfg.jetblock_space.get('conv_ks', [7]):
                    for g in self.cfg.jetblock_space.get('gate', ['token']):
                        yield {'full_layers': seed, 'jetblock': {'feature_map':f, 'conv_ks':ks, 'gate':g}}

    def apply_candidate(self, cand):
        for i,l in enumerate(self.model.layers):
            if i not in cand['full_layers']:
                l.attn = make_jetblock_like(l.attn, cand['jetblock'])

    def search(self, budget: int=32):
        self.freeze_mlps()
        best = None
        for i, cand in enumerate(self.candidate_architectures()):
            if i >= budget: break
            self.apply_candidate(cand)
            autotune_model(self.model, self.cfg.schedule_space)
            scores = self.eval_fn(self.model)
            cand['scores'] = scores
            score = 0.5 * scores.get('throughput', 0.0) + 0.5 * scores.get('mmlu', 0.0)
            if (best is None) or (score > (0.5*best['scores'].get('throughput',0.0)+0.5*best['scores'].get('mmlu',0.0))):
                best = cand
        return best

def make_jetblock_like(attn_layer, jb_cfg_dict): ...
def autotune_model(model, sched_space): ...
