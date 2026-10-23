def train_sapinn(net, x_f, x_ic, u_ic, x_bc):
    lam_f = torch.zeros(N_F, 1, device=DEVICE, requires_grad=True)
    lam_ic = torch.zeros(N_IC, 1, device=DEVICE, requires_grad=True)
    lam_bc = torch.zeros(N_BC, 1, device=DEVICE, requires_grad=True)
    sp = nn.functional.softplus                      # keeps weights positive
    opt_net = torch.optim.Adam(net.parameters(), lr=LR)
    opt_lam = torch.optim.Adam([lam_f, lam_ic, lam_bc], lr=LR)
    for _ in range(EPOCHS):
        r, _ = residual(net, x_f)
        e_ic = (net(x_ic) - u_ic) ** 2
        e_bc = net(x_bc) ** 2
        loss = (torch.mean(sp(lam_f) * r ** 2)
                + torch.mean(sp(lam_ic) * e_ic)
                + torch.mean(sp(lam_bc) * e_bc))
        opt_net.zero_grad()
        opt_lam.zero_grad()
        loss.backward()
        for lam in (lam_f, lam_ic, lam_bc):         
            lam.grad.neg_()
        opt_net.step()
        opt_lam.step()
    return net
