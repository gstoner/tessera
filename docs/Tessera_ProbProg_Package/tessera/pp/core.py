
# tessera/pp/core.py
import functools

def model(fn):
    fn._is_model = True
    return fn

def sample(name, dist, reparam=True):
    return dist.sample()

def observe(name, dist, value):
    return dist.log_prob(value)
