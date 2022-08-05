import autograd
import jax


x = autograd.Variable(2)
y = autograd.Variable(4)

mul_operation = autograd.multiply(x, y)

print("Forward result:", mul_operation.forward())
# output: Forward result: 8

print("Backward result with respect to x:", mul_operation.compute_gradients(with_respect=x))
# output: Backward result with respect to x: 4.0

print("Backward result with respect to y:", mul_operation.compute_gradients(with_respect=y))
# output: Backward result with respect to y: 2.0

# JAX implementation
def multiply(x, y):
    return x * y

print("JAX Forward result:", multiply(2., 4.))
# output: JAX Forward result: 8.0

# argnums is the same as with_respect in autograd,
# it means which parameter you want to differetiate with respect to it.
# here we differentiate with respect to X by setting argnums=0 
# which means the first function argument `X`.
print("JAX backward result with respect to x:", jax.grad(multiply, argnums=0)(2., 4.))
# output: JAX backward result with respect to x: 4.0


# argnums = 1 means with respect  to `y``.
print("JAX backward result with respect to y:", jax.grad(multiply, argnums=1)(2., 4.))
# output: JAX backward result with respect to y: 2.0

# If you have a multiple operations like the following
# Variables
x = autograd.Variable(2.)
y = autograd.Variable(3.)
z = autograd.Variable(10.)
# Operations
add_op = autograd.add(x, y)
mul_op = autograd.multiply(add_op, z)
pow_op = autograd.power(mul_op, 2)

print("Forward result:", pow_op.forward())
# output: Forward result: 2500.0

# To get the gradients we will call `compute_gradients` from the last operation `pow_op`
print("Backward result with respect to x:", pow_op.compute_gradients(with_respect=x))
# output: Backward result with respect to x: 1000.0

print("Backward result with respect to y:", pow_op.compute_gradients(with_respect=y))
# output: Backward result with respect to y: 1000.0

print("Backward result with respect to z:", pow_op.compute_gradients(with_respect=z))
# output: Backward result with respect to z: 500.0


# In JAX
def fun(x, y, z):
    out = x + y
    out = out * z
    out = out ** 2
    return out


print("JAX Forward result:", fun(2., 3., 10.))
# output: JAX Forward result: 2500.0

# To get the gradients we will call `jax.grad`
print("JAX Backward result with respect to x:", jax.grad(fun, argnums=0)(2., 3., 10.))
# output: JAX Backward result with respect to x: 1000.0

print("JAX Backward result with respect to y:", jax.grad(fun, argnums=1)(2., 3., 10.))
# output: JAX Backward result with respect to y: 1000.0

print("JAX Backward result with respect to z:", jax.grad(fun, argnums=2)(2., 3., 10.))
# output: JAX Backward result with respect to z: 500.0


x = autograd.Variable(0.2)

# sigmoid operation
sigmoid_op = autograd.sigmoid(x)
print("Forward result: ", sigmoid_op.forward())
# output: Forward result:  0.549833997312478

print("Backward result with respect to x:", sigmoid_op.compute_gradients(x))
# output: Backward result with respect to x: 0.24751657271185995

# In JAX
def sigmoid(x):
    exp_result = jax.numpy.exp(-x)
    return 1 / (1 + exp_result)


print("JAX forward result:", sigmoid(0.2))
# output: Backward result with respect to x: 0.24751657271185995

# argnums = 0 means with respect  to `x``.
print("JAX backward result with respect to x:", jax.grad(sigmoid, argnums=0)(0.2))
# output: JAX backward result with respect to x: 0.24751654
