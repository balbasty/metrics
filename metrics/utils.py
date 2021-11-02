import itertools


def broadcastable(x, y):
    xshape = x.shape[-min(x.ndim, y.ndim):]
    yshape = y.shape[-min(x.ndim, y.ndim):]
    return all(sx == sy or sx == 1 or sy == 1 for sx, sy in zip(xshape, yshape))


def check_broadcastable(*arrays):
    for x, y in itertools.combinations(arrays, 2):
        if not broadcastable(x, y):
            raise ValueError(f'Expected inputs to have broadcastable shapes '
                             f'but got {x.shape} and {y.shape} which are not.')