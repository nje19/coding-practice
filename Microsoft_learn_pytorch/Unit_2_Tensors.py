#%%
import torch
import numpy as np

# creating tensors directly from data
data = [[1,2],[3,4]]
xdata = torch.tensor(data)

# creating tensors from numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# print(f'Numpy np_array value: \n {np_array} \n')
# print(f'Tensor x_np value: \n {x_np} \n')

# if we change the numpy array, the tensor created from this will also change as they have the same memory location
# need to either specify output as original array or another predefined array, or need to equal np.multiply to something, otherwise it won't know where to store the new array
np.multiply(np_array, 2, out= np_array)

# creating a tensor from another tensor
# rand_like expects input tensor to have dataype of float, so if this isn't the case, need to specify output tensor to have this, or change input tensor earlier on to be a float, using x_np.float() for example
x_ones = torch.ones_like(x_np)
x_rand = torch.rand_like(x_np, dtype=torch.float)
x_zeros = torch.zeros_like(x_np)

# creating tensors by specifying the shape of the tensor
# shape is specified in the order of rows, then columns
shape = (4,3)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# displaying attributes of a tensor

tensor = torch.rand((4,3))
# print(tensor)
# print(f'Tensor shape: {tensor.shape}')
# print(f'Tensor data type: {tensor.dtype}')
# print(f'Device tensor is stored on: {tensor.device}')

# moving a tensor from cpu to gpu if available
tensor_move = torch.tensor([[2,3],[4,5]])
if torch.cuda.is_available():
    tensor_move = tensor.to('cuda')
# print(tensor_move.device)
# gpu not available so still stored on cpu

# practicing numpy indexing and slicing with tensors
tensor = torch.rand(4,3)
# print(tensor)
# print(f'First row: {tensor[0]}')
# print(f'First column: {tensor[:,0]}')
# print(f'Last column: {tensor[:,-1]}')
# changing middle column to all zeros
tensor[:, 1] = 0
# print(tensor)

# to join tensors, can use torch.cat (concatenates tensors along given dimension) or torch.stack (concatenates tensors along new dimension)
# cat joins the tensors together, whereas stack keeps them individual but adds them both to a new tensor in a list
tensor_x = torch.rand(3,3)
# print(f'Original tensor:  \n {tensor_x} \n')
t1 = torch.cat((tensor_x,tensor_x), dim = 0)
t2 = torch.stack((tensor_x,tensor_x), dim = 0)
# print(t1)
# print()
# print(t2)

# 3 different ways/notations to perform matrix multiplication between two tensors
y1 = tensor_x @ tensor_x.T # @ is the matrix multiplicaiton sign, and T is the transform tensor
y2 = tensor_x.matmul(tensor_x.T)
y3 = torch.matmul(tensor_x, tensor_x.T)
# print(y1)
# print(y2)
# print(y3)

# 3 different ways to compute element-wise product of two matrices]
z1 = tensor_x * tensor_x
z2 = tensor_x.mul(tensor_x)
z3 = torch.mul(tensor_x, tensor_x)
# print(z1)
# print(z2)
# print(z3)

# .sum() adds up all the values in the tensor
# to convert a single-elemnent tensor into a number, use .item()
agg = tensor_x.sum()
agg_item = agg.item()
# print(agg)
# print(agg_item)

