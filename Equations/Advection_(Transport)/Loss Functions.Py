def residual_pointwise(net, xt, m):
    """Advection residual r = u_t + c*u_x."""
    xt = xt.clone().requires_grad_(True)
    u = net(xt)
    grad_u = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    r = u_t + C * u_x
    return r, u_x


def condition_loss(net, x_ic, x_left, x_right, m):
    """Combined IC + periodic boundary loss, treated as one condition term."""
    x = x_ic[:, 0:1]
    u_ic_target = torch.sin(m * np.pi * x)
    L_ic = torch.mean((net(x_ic) - u_ic_target) ** 2)
    L_per = torch.mean((net(x_left) - net(x_right)) ** 2)
    return L_ic + L_per


def all_losses(net, x_f, x_ic, x_left, x_right, m):
    r, _ = residual_pointwise(net, x_f, m)
    L_f = torch.mean(r ** 2)
    L_bc = condition_loss(net, x_ic, x_left, x_right, m)
    return L_f, L_bc
