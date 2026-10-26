def residual_pointwise(net, xt, return_ux=False):
    """Pointwise Fisher-KPP residual  r = u_t - D u_xx - rho u(1-u).
       Returns r:(N,1).   If return_ux, also returns u_x for GW-PINN."""
    xt = xt.clone().requires_grad_(True)
    u  = net(xt)
    ones = torch.ones_like(u)
    g  = torch.autograd.grad(u, xt, grad_outputs=ones, create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    r = u_t - D_COEFF * u_xx - RHO * u * (1.0 - u)
    if return_ux:
        return r, u_x
    return r


def ic_loss(net, x_ic, u_ic_target):
    """MSE of the initial condition."""
    return torch.mean((net(x_ic) - u_ic_target) ** 2)


def bc_loss(net, x_bc_left, x_bc_right):
    """Dirichlet u(0,t)=1  +  Neumann u_x(L,t)=0."""
    # left: Dirichlet
    L_dir = torch.mean((net(x_bc_left) - 1.0) ** 2)
    # right: Neumann  u_x = 0
    xr = x_bc_right.clone().requires_grad_(True)
    u_r = net(xr)
    gr = torch.autograd.grad(u_r, xr, grad_outputs=torch.ones_like(u_r),
                             create_graph=True)[0]
    u_x_R = gr[:, 0:1]
    L_neu = torch.mean(u_x_R ** 2)
    return L_dir + L_neu
