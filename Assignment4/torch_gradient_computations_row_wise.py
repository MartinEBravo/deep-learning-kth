import torch
import numpy as np


# assumes X has size n x K, h0 has size 1 x K, etc
def ComputeGradsWithTorch(X, y, h0, RNN):
    tau = X.shape[1]

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)

    ## give informative names to these torch classes
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)

    # create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(X.shape[0], X.shape[1], dtype=torch.float64)

    hprev = ht
    for t in range(tau):
        #### BEGIN your code ######

        # Code to apply the RNN to hprev and Xt[t:t+1, :] to compute the hidden scores "Hs" at timestep t
        # (ie equations (1,2) in the assignment instructions)
        # Store results in Hs

        # Don't forget to update hprev!
        hprev = apply_tanh(
            torch_network["W"] @ hprev
            + torch_network["U"] @ Xt[t : t + 1, :]
            + torch_network["b"]
        )
        Hs[t] = hprev

        #### END of your code ######

    Os = torch.matmul(Hs, torch_network["V"]) + torch_network["c"]
    P = apply_softmax(Os)

    # compute the loss
    loss = torch.mean(
        -torch.log(P[np.arange(tau), y])
    )  # use this line if storing inputs row-wise

    # compute the backward pass relative to the loss and the named parameters
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads
