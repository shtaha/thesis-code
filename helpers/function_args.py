def dual_fn(a, b):
    return a + b


def wrapper_a_fn(a):
    return lambda b: dual_fn(a, b)

a = 3
b = 1

singular_fn = wrapper_a_fn(a)

print(dual_fn(a, b))
print(singular_fn(b))
print(singular_fn(3))
print(singular_fn(2))
print(singular_fn(6))
