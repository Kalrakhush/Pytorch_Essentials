#calculating gradient using autograd
import torch

x=torch.randn(3, requires_grad=True)
print(x)

y=x+2  #here requires_gard will automatically generate backward pass function 
print(y)

#Output: tensor([1.3753, 2.6947, 1.8910], grad_fn=<AddBackward0>)

z=y*y*2
print(z)

#Output: tensor([ 8.0000, 14.5000,  7.1000], grad_fn=<MulBackward0>)

z=z.mean()
print(z)

#Output: tensor(9.5333, grad_fn=<MeanBackward0>)

z.backward() #dz/dx
print(x.grad)

#if we want pytorch not to track the gradient
#we can use torch.no_grad() context manager
with torch.no_grad():
    y=x+2
    print(y)

#whenever we call a backward function, the gradient will be accumulates, values will be added to the existing gradient
# if we want to reset the gradient, we can use zero_grad() function
# for example
# x.grad.zero_()
# x.grad.data.zero_()
# x.grad=None
# x.grad.detach_()
   
weights=torch.ones(3, requires_grad=True) 

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward() # dz/dx (calcualtes the gradient)
    print(weights.grad) # dz/dw (gradient of the model output with respect to the weights)
    #output:
    # tensor([3., 3., 3.])
    # tensor([6., 6., 6.])
    # tensor([9., 9., 9.])
    weights.grad.zero_() # reset the gradient
    print(weights.grad)

