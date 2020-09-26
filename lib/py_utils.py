def wrapper(b, dual_fn):
    return lambda a: dual_fn(a, b)
