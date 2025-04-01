import torch

def ComputeGradsWithTorch(X, y, network_params):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)    
    
    N = X.shape[1]
    
    scores = torch.matmul(W, Xt)  + b;

    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)
    
    ## compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    

    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()

    return grads    
