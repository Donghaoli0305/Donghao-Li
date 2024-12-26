"""
This file and its tests are individualized for NetID dli106.
"""
import numpy as np
import torch as tr

def mountain1d(x):
    """
    Input: float x
    Output: floats z, dz_dx
      z: value of -3*x**3 + 1*x**2 + 1*x + 3
      dz_dx: value of dz/dx evaluated at input
    Use torch to compute derivatives and then type-cast back to float
    """
    x_tensor = tr.tensor(x, requires_grad=True)
    

    z = -3 * x_tensor**3 + x_tensor**2 + x_tensor + 3    

    z.backward() 
    
    z_value = z.item()
    dz_dx= x_tensor.grad.item()
    
    return z_value, dz_dx

def robot(t1, t2):
    """
    Input: floats t1, t2
        each joint angle, in units of radians
    Output: floats z, dz_dt1, dz_dt2
      z: value of (x - -3)**2 + (y - -3)**2, where
         x = 2*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) + 3*cos(t1)
         y = 2*(sin(t1)*cos(t2) + cos(t1)*sin(t2)) + 3*sin(t1)
      dz_dt1, dz_dt2: values of dz/dt1 and dz/dt2 evaluated at input
    Use torch to compute derivatives and then type-cast back to float
    """
    t1_tensor = tr.tensor(t1, requires_grad=True)
    t2_tensor = tr.tensor(t2, requires_grad=True)
    
    x = 2 * (tr.cos(t1_tensor) * tr.cos(t2_tensor) - tr.sin(t1_tensor) * tr.sin(t2_tensor)) + 3 * tr.cos(t1_tensor)
    y = 2 * (tr.sin(t1_tensor) * tr.cos(t2_tensor) + tr.cos(t1_tensor) * tr.sin(t2_tensor)) + 3 * tr.sin(t1_tensor)
    
    z = (x - (-3))**2 + (y - (-3))**2

    z.backward()
    
    z_value = z.item()
    dz_dt1 = t1_tensor.grad.item()
    dz_dt2 = t2_tensor.grad.item()
    
    return z_value, dz_dt1, dz_dt2

def neural_network(W1, W2, W3):
    """
    Input: numpy arrays W1, W2, and W3 representing weight matrices
    Output: y, e, de_dW1, de_dW2, and de_dW3
        float y: the output of the neural network
        float e: the squared error of the neural network
        numpy array de_dWk: the gradient of e with respect to Wk, for k in [1, 2, 3]
    Use torch to compute derivatives and then type-cast back to floats and numpy arrays
    The following documentation may be helpful:
        https://pytorch.org/docs/stable/generated/torch.tanh.html
        https://pytorch.org/docs/stable/generated/torch.mv.html
        https://pytorch.org/docs/stable/tensors.html
        https://pytorch.org/docs/stable/generated/torch.Tensor.float.html#torch.Tensor.float
        https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html#torch.Tensor.numpy
        https://numpy.org/doc/stable/user/basics.types.html
    For more information, consult the instructions.
    """
    
    W1_t = tr.tensor(W1, requires_grad=True, dtype=tr.float32)
    W2_t = tr.tensor(W2, requires_grad=True, dtype=tr.float32)
    W3_t = tr.tensor(W3, requires_grad=True, dtype=tr.float32)
    
    x1 = tr.tensor([-1.0, 1.0], dtype=tr.float32) 
    x2 = tr.tensor([1.0, 1.0, 1.0], dtype=tr.float32) 
    x3 = tr.tensor([-1.0, -1.0, -1.0, -1.0], dtype=tr.float32)  
    

    h3 = tr.tanh(x3) 
    h2 = tr.tanh(x2 + tr.mv(W3_t, h3))
    h1 = tr.tanh(x1 + tr.mv(W2_t, h2)) 
    y = tr.mv(W1_t, h1)

    e = (y - 1.0)**2 
    
    e.backward()
    
    y_val = y.item()
    e_val = e.item()
    de_dW1 = W1_t.grad.numpy()
    de_dW2 = W2_t.grad.numpy()
    de_dW3 = W3_t.grad.numpy()
    
    return y_val, e_val, de_dW1, de_dW2, de_dW3

if __name__ == "__main__":

    # start with small random weights
    W1 = np.random.randn(1,2).astype(np.float32) * 0.01
    W2 = np.random.randn(2,3).astype(np.float32) * 0.01
    W3 = np.random.randn(3,4).astype(np.float32) * 0.01
    
    # do several iterations of gradient descent
    for step in range(100):
        
        # evaluate loss and gradients
        y, e, dW1, dW2, dW3 = neural_network(W1, W2, W3)
        if step % 10 == 0: print("%d: error = %f" % (step, e))

        # take step
        eta = .1/(step + 1)
        W1 -= dW1 * eta
        W2 -= dW2 * eta
        W3 -= dW3 * eta

