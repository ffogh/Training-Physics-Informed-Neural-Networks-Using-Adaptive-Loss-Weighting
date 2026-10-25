
def train_sapinn(net, x_f, x_ic, u_ic, x_bc):

    # 1. Initialize self-adaptive weights
    col_weights = torch.full(
        (x_f.shape[0], 1),
        100.0,
        device=DEVICE,
        requires_grad=True
    )

    u_weights = torch.rand(
        (x_ic.shape[0], 1),
        device=DEVICE,
        requires_grad=True
    )

    # 2. Initialize optimizers
    opt_net = torch.optim.Adam(net.parameters(), lr=LR)

    opt_col = torch.optim.Adam([col_weights], lr=LR)
    opt_u = torch.optim.Adam([u_weights], lr=LR)

    # 3. Training
    for _ in range(EPOCHS):

        # Calculate PDE residual
        r, _ = residual(net, x_f)

        # Calculate IC error
        e_ic = net(x_ic) - u_ic

        # Calculate BC error (zero Dirichlet BC)
        e_bc = net(x_bc)

        # 4. Apply self-adaptive weights
        mse_f = torch.mean((col_weights * r) ** 2)

        mse_ic = torch.mean((u_weights * e_ic) ** 2)

        # Original implementation: BC has no adaptive weights
        mse_bc = torch.mean(e_bc ** 2)

        # Total loss
        loss = mse_f + mse_ic + mse_bc

        # 5. Reset gradients
        opt_net.zero_grad()
        opt_col.zero_grad()
        opt_u.zero_grad()

        # 6. Calculate gradients
        loss.backward()

        # 7. Gradient ascent for adaptive weights
        col_weights.grad.neg_()
        u_weights.grad.neg_()

        # 8. Gradient descent for network
        opt_net.step()

        # 9. Gradient ascent for adaptive weights
        opt_col.step()
        opt_u.step()

    return net






For a fair comparison across 13 PDE benchmarks, we used a unified Adam-only training protocol for all methods. Although the original SA-PINN paper employed 
L-BFGS refinement in some experiments, it was not used universally. We therefore omitted L-BFGS for all 
methods to maintain consistent optimization conditions and computational budgets while preserving their core adaptive weighting mechanisms. The following is from the paper 

''''''''In some case, these are followed by another fixed number of iterations of the L-BFGS quasi-newton method.''''''
