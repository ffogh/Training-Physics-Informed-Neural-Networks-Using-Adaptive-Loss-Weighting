def residual_pointwise(net, xt, c):
    """Pointwise wave residual r = u_tt - c^2 u_xx  at xt:(N,2).
       Returns r:(N,1) and u_x:(N,1) (u_x is the 1-D gradient for GW-PINN)."""
    xt = xt.clone().requires_grad_(True)
    u = net(xt)
    ones = torch.ones_like(u)
    u_g = torch.autograd.grad(u, xt, grad_outputs=ones, create_graph=True)[0]
    u_x, u_t = u_g[:, 0:1], u_g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, xt, grad_outputs=torch.ones_like(u_t),
                               create_graph=True)[0][:, 1:2]
    r = u_tt - (c ** 2) * u_xx
    return r, u_x


def ic_disp_loss(net, x_ic, u_ic):
    """MSE of the initial displacement u(x,0) = sin(pi x)."""
    return torch.mean((net(x_ic) - u_ic) ** 2)


def ic_vel_loss(net, x_ic):
    """MSE of the initial velocity u_t(x,0) = 0."""
    x_ic = x_ic.clone().requires_grad_(True)
    u = net(x_ic)
    u_t = torch.autograd.grad(u, x_ic, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0][:, 1:2]
    return torch.mean(u_t ** 2)


def bc_loss(net, x_bc):
    """MSE of the homogeneous Dirichlet boundary u(0,t) = u(1,t) = 0."""
    return torch.mean(net(x_bc) ** 2)


def all_losses(net, x_f, x_ic, u_ic, x_bc, c):
    """Return the four scalar losses L_f, L_ic, L_ict, L_bc."""
    r, _ = residual_pointwise(net, x_f, c)
    L_f   = torch.mean(r ** 2)
    L_ic  = ic_disp_loss(net, x_ic, u_ic)
    L_ict = ic_vel_loss(net, x_ic)
    L_bc  = bc_loss(net, x_bc)
    return L_f, L_ic, L_ict, L_bc
