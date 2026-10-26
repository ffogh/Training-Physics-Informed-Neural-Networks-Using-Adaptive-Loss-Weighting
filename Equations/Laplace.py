def residual_pointwise(net, xy):
    """Pointwise Laplace residual r = u_xx + u_yy  (source-free) at xy:(N,2).
       Returns r:(N,1) and the divergence surrogate (u_x+u_y) for GW-PINN."""
    xy = xy.clone().requires_grad_(True)
    u = net(xy)
    ones = torch.ones_like(u)
    g = torch.autograd.grad(u, xy, grad_outputs=ones, create_graph=True)[0]
    u_x, u_y = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:, 1:2]
    r = u_xx + u_yy                                # Delta u = 0
    div_surrogate = u_x + u_y                      # 2-D divergence surrogate
    return r, div_surrogate


def bc_loss(net, x_bc, u_bc):
    """MSE of the (non-homogeneous) Dirichlet boundary  u = g."""
    return torch.mean((net(x_bc) - u_bc) ** 2)


def all_losses(net, x_f, x_bc, u_bc):
    """Return the two scalar losses L_f, L_bc."""
    r, _ = residual_pointwise(net, x_f)
    L_f  = torch.mean(r ** 2)
    L_bc = bc_loss(net, x_bc, u_bc)
    return L_f, L_bc
