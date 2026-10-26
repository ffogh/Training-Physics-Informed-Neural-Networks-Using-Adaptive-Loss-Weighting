def train_gwpinn(net, x_f, x_ic, u_ic, x_bc):


    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for _ in range(EPOCHS):
        opt.zero_grad()

        r, d = residual(net, x_f)          # r: PDE residual, d: u_x + u_y

        # Gradient weight, treated as a constant (detached) at each iteration
        w = (1.0 / (GW_EPS2 * (torch.abs(d) - d) + 1.0)).detach()

        loss = (
            torch.mean(w * r ** 2)         # weight applied once (Kapoor et al., Eq. 3)
            + ic_loss(net, x_ic, u_ic)     # epsilon_1 = 1 (no extra data weight)
            + bc_loss(net, x_bc)
        )

        loss.backward()
        opt.step()
    return net
    
# Detachment was tested in some cases, but better results were obtained without it.
We did not find any codes for this method. 

For a fair and computationally consistent comparison, we adapted the methods to a unified experimental framework while accounting for their computational costs.  
All methods were trained using the Adam optimizer under identical network architectures, training budgets, and experimental conditions.  
We adopted the default hyperparameters reported in the original papers for all methods, without performing any additional hyperparameter tuning.
