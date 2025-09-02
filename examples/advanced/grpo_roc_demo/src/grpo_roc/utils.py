
import random, torch
def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)

def pack_traj_logprob(logits, tokens):
    import torch
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, tokens.view(-1,1)).sum().squeeze()
