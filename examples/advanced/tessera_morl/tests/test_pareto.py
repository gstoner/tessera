import numpy as np
from python.morl_utils import pareto_mask

def test_basic():
    pts = np.array([[0,0],[1,0],[0,1],[0.5,0.5],[2,2],[1.5,1.5]], dtype=np.float32)
    mask = pareto_mask(pts)
    # Points [2,2] and [1.5,1.5] dominate many; [0.5,0.5] is dominated by both.
    assert mask.sum() == 2
    assert mask[-1] == False
