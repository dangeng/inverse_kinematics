import torch

'''
Torch allows us to compute derivatives by building a computational graph:
'''

# Make a tensor, and tell torch to keep track of gradients
x = torch.tensor(2., requires_grad=True)

# Do an operation of x (y = x**2)
y = x**2

# Calculate gradients
y.backward()

# Print out gradient of y with respect to x
# We should have dy/dx = 2x, and with x = 2, dy/dx = 4
print('Gradient of x**2 wrt x, evaluated at x=2 is: ')
print(x.grad, '\n')




'''
We can even do fancy things like differentiating a matrix
'''

# Make a 2x2 tensor, tell torch to make computational graph
X = torch.tensor([[1.,2],[3,4]], requires_grad=True)

# Do operation on X
y = X.sum()

# Calculate gradients
y.backward()

# Print out gradient of y with respect to X
# We should have y = sum(X) = X_11 + X_12 + X_21 + X_22
# So dy/dX = [[dy/dX_11, dy/dX_12], [dy/dX_21, dy/dX_22]] =  2x2 matrix of 1's
print('Gradient of sum(X) wrt each element of X, evaluated at x=X is: ')
print(X.grad, '\n')




'''
In solving the inverse kinematic problem we have the following setup:

    1. Sample x
    2. Pass x through NN to predict thetas
    3. Pass theta through forward kinematics model to get x'
    4. Calculate loss as L = MSE(x, x')
    5. Differentiate L wrt parameters of NN
    6. Update NN parameters with SGD, using gradients we calculated in step 5

However, in order for this to work we need to be able to differentiate L wrt the NN params
This only works if each operation we apply is differentiable. In particular
we would like the "forward kinematic" operation in step 3 to be differentiable

Forward kinematics is just a series of matrix multiplies and cosine and sines,
so in theory it should be differentiable but we have to be careful in practice.
In constructing the DH matrices it's possible to do it incorrectly and end
up with something that isn't differentiable to pytorch (that is, something
that doesn't "build" the computational graph)

As a simple example, the naive way to make a DH matrix could look something like this:
'''

# Make theta, tell torch we want it to track gradients / build a computational graph
theta = torch.tensor(2., requires_grad=True)

# Construct the "DH matrix" -> 2x2 matrix filled with theta
dh = torch.tensor([[theta, theta], [theta, theta]])

# Calculate some value from the "DH matrix"
# y = sum(DH) = theta + theta + theta + theta = 4 * theta
y = dh.sum()

# Try to differentiate
# We should have dy/d(theta) = d/d(theta) (4 * theta) = 4
try:
    y.backward()
except:
    print("Differentiating this object doesn't work, because it wasn't constructed properly!")
    print("The problem is when we write `dh = torch.tensor(...`, we copy the value of theta, but not it's computational graph information")
    print("As a result dh doesn't have a `grad_fn` and it's `requires_grad` attr is False: ")
    print('\tdh.grad_fn: ', str(dh.grad_fn))
    print('\tdh.requires_grad: ', dh.requires_grad)
    print('\n')

'''
Instead if we copy the values, we get a dh matrix that we can "differentiate through"
'''

# Make theta, tell torch we want it to track gradients / build a computational graph
theta = torch.tensor(2., requires_grad=True)

# Construct a tensor to hold the "DH matrix"
dh = torch.zeros(2,2)

# Populate the DH matrix with values -> 2x2 matrix filled with theta
dh[0,0] = theta
dh[1,0] = theta
dh[0,1] = theta
dh[1,1] = theta

print('Now the "DH Matrix" is part of the computational graph: ')
print('\tdh.grad_fn: ', str(dh.grad_fn))
print('\tdh.requires_grad: ', dh.requires_grad)

# Calculate some value from the "DH matrix"
# y = sum(DH) = theta + theta + theta + theta = 4 * theta
y = dh.sum()

# Differentiate
y.backward()

# Print out gradients. We should have dy/d(theta) = d/d(theta) (4 * theta) = 4
print('Gradient of y=sum(DH) wrt theta, evaluated at theta=2 is: ')
print(theta.grad, '\n')

'''
So in summary, just building the DH matrix naively erases gradient information.
As a result when we try to differentiate the loss function we will get an error.
Building the DH matrix in this awkward way preserves the gradient information
(it maintains a computational graph), and we can differentiate and update the NN params

Tbh, there might be a cleaner way to construct the DH matrix and preserve autodiff
info. This was just the first thing that came to mind.
'''
