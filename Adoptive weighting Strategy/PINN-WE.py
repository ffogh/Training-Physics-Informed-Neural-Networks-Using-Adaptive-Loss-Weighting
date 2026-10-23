def train_gwpinn(net, x_f, x_ic, u_ic, x_bc):

    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for _ in range(EPOCHS):
        opt.zero_grad()

        r, d = residual(net, x_f)

        # Gradient weighting without detach
        w = 1.0 / (GW_EPS2 * (torch.abs(d) - d) + 1.0)

        loss = (
            torch.mean(w * r ** 2)
            + ic_loss(net, x_ic, u_ic)
            + bc_loss(net, x_bc)
        )

        loss.backward()
        opt.step()

    return net

# Detachment was tested in some cases, but better results were obtained without it.
