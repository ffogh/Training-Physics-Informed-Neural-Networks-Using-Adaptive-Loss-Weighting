def train_ipinn(net, x_f, x_bc, u_bc):
    Q = torch.zeros(2, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam(list(net.parameters()) + [Q], lr=LR)
    for ep in range(EPOCHS):
        opt.zero_grad()
        terms = torch.stack(all_losses(net, x_f, x_bc, u_bc))
        loss = (torch.exp(-Q) * terms).sum() + Q.sum()
        loss.backward()
        opt.step()
    return net


For a fair comparison across 13 PDE benchmarks, we implemented I-PINN using its Gaussian-likelihood-based adaptive weighting mechanism. 
We did not incorporate the improved neural network architecture proposed in the original paper, as our objective 
was to compare adaptive loss weighting methods independently of architectural modifications. 
All methods therefore employed the same network architecture and Adam-only optimization protocol to ensure consistent experimental conditions.
