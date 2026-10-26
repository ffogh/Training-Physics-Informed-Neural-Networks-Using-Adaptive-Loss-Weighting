def train_sapinn(net, x_f, x_ic, u_ic, x_bc):
    """SA-PINN (McClenny & Braga-Neto): pointwise self-adaptive weights,
    minimised w.r.t. the network and maximised w.r.t. the weights."""

    # 1. Initialize self-adaptive weights (as in the original code)
    # Residual weights: random in [0, 1)        <- tf.random.uniform([N_f, 1])
    col_weights = torch.rand(
        (x_f.shape[0], 1),
        device=DEVICE,
        requires_grad=True
    )

    # IC weights: 100 x random in [0, 1)        <- 100 * tf.random.uniform([N0, 1])
    u_weights = (100.0 * torch.rand(
        (x_ic.shape[0], 1),
        device=DEVICE
    )).requires_grad_(True)

    # 2. Initialize optimizers
    opt_net = torch.optim.Adam(net.parameters(), lr=LR)
    opt_col = torch.optim.Adam([col_weights], lr=LR)
    opt_u = torch.optim.Adam([u_weights], lr=LR)

    # 3. Training
    for _ in range(EPOCHS):

        # PDE residual
        r, _ = residual(net, x_f)

        # IC error
        e_ic = net(x_ic) - u_ic

        # BC error (zero Dirichlet BC)
        e_bc = net(x_bc)

        # 4. Apply self-adaptive weights
        mse_f = torch.mean((col_weights * r) ** 2)
        mse_ic = torch.mean((u_weights * e_ic) ** 2)

        # Original implementation: BC has no adaptive weights
        mse_bc = torch.mean(e_bc ** 2)

        loss = mse_f + mse_ic + mse_bc

        # 5. Reset gradients
        opt_net.zero_grad()
        opt_col.zero_grad()
        opt_u.zero_grad()

        # 6. Compute gradients
        loss.backward()

        # 7. Flip sign for gradient ascent on the adaptive weights
        col_weights.grad.neg_()
        u_weights.grad.neg_()

        # 8. Gradient descent for the network
        opt_net.step()

        # 9. Gradient ascent for the adaptive weights
        opt_col.step()
        opt_u.step()

    return net





For a fair comparison across 13 PDE benchmarks, we used a unified Adam-only training protocol for all methods. Although the original SA-PINN paper employed 
L-BFGS refinement in some experiments, it was not used universally. We therefore omitted L-BFGS for all 
methods to maintain consistent optimization conditions and computational budgets while preserving their core adaptive weighting mechanisms. The following is from the paper 

''''''''In some case, these are followed by another fixed number of iterations of the L-BFGS quasi-newton method.''''''

SA-PINN improves training on challenging PDEs, such as Allen–Cahn.
The paper mensioned “In our implementation of SA-PINNs, we use Tensorflow

For a fair and computationally consistent comparison, we adapted the methods to a unified experimental framework while accounting for their computational costs.  
All methods were trained using the Adam optimizer under identical network architectures, training budgets, and experimental conditions.  
We adopted the default hyperparameters reported in the original papers for all methods, without performing any additional hyperparameter tuning.
