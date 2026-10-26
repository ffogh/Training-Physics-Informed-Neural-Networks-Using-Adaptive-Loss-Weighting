

def _ntk_trace(outputs, params):
    trace = 0.0
    for out in outputs:
        grads = torch.autograd.grad(out, params, retain_graph=True, allow_unused=True)
        trace += sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return trace


def train_pipnns_ntk(net, x_f, x_ic, u_ic, x_bc):
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    beta = torch.ones(3, device=DEVICE)

    for ep in range(EPOCHS):
        if ep % NTK_FREQ == 0:
            idx_f = torch.randperm(N_F, device=DEVICE)[:N_NTK]
            idx_ic = torch.randperm(N_IC, device=DEVICE)[:N_NTK]
            idx_bc = torch.randperm(N_BC, device=DEVICE)[:N_NTK]

            # Per-point errors of each loss term
            r, _ = residual(net, x_f[idx_f])
            e_f = r.flatten()
            e_ic = (net(x_ic[idx_ic]) - u_ic[idx_ic]).flatten()
            e_bc = net(x_bc[idx_bc]).flatten()

            errors = (e_f, e_ic, e_bc)
            n = torch.tensor([float(e.numel()) for e in errors], device=DEVICE)
            tr = torch.stack([_ntk_trace(e.unbind(), params) for e in errors])
            beta = ((n / n.sum()) * (tr.sum() / (tr + 1e-12))).detach()

        opt.zero_grad()
        terms = torch.stack(all_losses(net, x_f, x_ic, u_ic, x_bc))
        (beta * terms).sum().backward()
        opt.step()
    return net







For PIPNNs-NTK, we adopt only its NTK-based loss-weighting rule and apply it to a single network, since our benchmark is a single-domain 
forward problem; domain decomposition, sensor-data and interface losses, and parameter identification are therefore omitted, so that all 
methods share the same network and optimizer. Because computing the NTK requires one backward pass per training point, evaluating it over 
all 6,144 training points at every iteration would cost roughly 6×10⁷ additional backward passes, about 6,000 times the training cost of a 
standard PINN. Following Wang et al. (2022), we therefore estimate each NTK block trace from 64 randomly sampled points per loss term and 
update the weights every 100 epochs, which reduces this overhead by a factor of about 3,200 while preserving the weighting rule
(For fairness and computational cost, PIPNNs-NTK is applied on a single domain (m = n = 1), so that all methods share the same network and training budget.)


For a fair and computationally consistent comparison, we adapted the methods to a unified experimental framework while accounting for their computational costs.  
All methods were trained using the Adam optimizer under identical network architectures, training budgets, and experimental conditions.  
We adopted the default hyperparameters reported in the original papers for all methods, without performing any additional hyperparameter tuning.


