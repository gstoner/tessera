
# tessera/pp/inference.py

class autoguide:
    @staticmethod
    def mean_field(model):
        return lambda *a, **kw: model(*a, **kw)

def svi(model, guide, data, steps=1000, optimizer="adam", seed=0):
    # Simplified skeleton
    losses = []
    for step in range(steps):
        loss = -guide(**data)
        losses.append(loss)
    return {"posterior": guide, "losses": losses}
